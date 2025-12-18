"""
pinecone_rag.py

Provides a high-level RAG helper that performs:
 - Dense embedding (Voyage-3-large for semantic meaning)
 - Sparse embedding (BM25/Pinecone for keyword matching)
 - Hybrid index query (dense + sparse in single Pinecone index)
 - Candidate merging (union of top-K)
 - Reranking via Anthropic LLM (or Cohere/Pinecone)

HYBRID SEARCH FLOW:
    User Query
         ↓
    Voyage Embedding (dense/meaning) + BM25 (sparse/keywords)
         ↓
    Dense Search      Sparse Search
         ↓                 ↓
         └──── Combine Results ────┘
                    ↓
                Reranker
                    ↓
               Top Documents

Usage:
    from app.services.pinecone_rag import combined_search
    results = await combined_search(query_text, index_name="research-hybrid")

"""
from __future__ import annotations

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "anthropic").lower()

# =========================================================
# VOYAGE CLIENT FOR DENSE EMBEDDINGS
# =========================================================
_voyage_client = None
try:
    import voyageai
    if VOYAGE_API_KEY:
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        logger.info("Voyage client initialized for dense embeddings (voyage-3-large)")
    else:
        logger.warning("VOYAGE_API_KEY not set - Voyage embeddings unavailable")
except ImportError:
    logger.warning("voyageai package not installed - dense embeddings will use fallback")

# =========================================================
# BM25 ENCODER FOR SPARSE EMBEDDINGS
# =========================================================
_bm25 = None
_BM25_AVAILABLE = False
try:
    from pinecone_text.sparse import BM25Encoder
    _bm25 = BM25Encoder().default()
    _BM25_AVAILABLE = True
    logger.info("BM25 encoder initialized for sparse embeddings")
except ImportError:
    logger.warning("pinecone_text package not installed - sparse embeddings will use fallback")
except Exception as e:
    logger.warning(f"BM25 encoder initialization failed: {e}")


def _get_pinecone_client():
    """Get or create Pinecone client."""
    try:
        from pinecone import Pinecone as PineconeClient
        return PineconeClient(api_key=PINECONE_API_KEY)
    except Exception:
        try:
            import pinecone as pc
            if PINECONE_API_KEY:
                env = os.getenv("PINECONE_INDEX_HOST")
                if env:
                    pc.init(api_key=PINECONE_API_KEY, environment=env)
                else:
                    pc.init(api_key=PINECONE_API_KEY)
            return pc
        except Exception as e:
            raise RuntimeError(f"Pinecone client not available: {e}")


async def _embed_query_voyage(query: str, model: str = "voyage-3-large") -> Optional[List[float]]:
    """
    Generate DENSE embedding using Voyage AI (voyage-3-large).
    
    Voyage embeddings capture:
    - Intent
    - Context  
    - Semantics (meaning)
    
    Args:
        query: Text to embed
        model: Voyage model (voyage-3-large is recommended for best quality)
    
    Returns:
        Dense embedding vector (list of floats)
    """
    if _voyage_client is None:
        logger.warning("Voyage client unavailable, falling back to embeddings.embed_text")
        from app.services.embeddings import embed_text
        vec = await asyncio.to_thread(embed_text, query)
        return vec.tolist() if hasattr(vec, 'tolist') else list(vec)
    
    try:
        def _embed():
            return _voyage_client.embed(
                texts=[query],
                model=model,
                input_type="query"  # Use "query" for search queries
            )
        
        response = await asyncio.to_thread(_embed)
        
        # Extract embedding from response
        embeddings = getattr(response, "embeddings", None)
        if embeddings and isinstance(embeddings, list) and embeddings:
            logger.debug(f"Voyage dense embedding: {len(embeddings[0])} dimensions")
            return embeddings[0]
        
        # Fallback
        from app.services.embeddings import embed_text
        vec = await asyncio.to_thread(embed_text, query)
        return vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        
    except Exception as e:
        logger.error(f"Voyage embedding failed: {e}")
        from app.services.embeddings import embed_text
        vec = await asyncio.to_thread(embed_text, query)
        return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def _encode_sparse_bm25(query: str) -> Optional[Dict[str, Any]]:
    """
    Generate SPARSE embedding using BM25.
    
    BM25 captures:
    - Exact keywords ("NVIDIA", "share", "price")
    - Important terms
    - Specific entities
    
    Args:
        query: Text to encode
    
    Returns:
        Sparse vector dict compatible with Pinecone
    """
    if _bm25 is None:
        # Fallback: simple token-based sparse vector
        tokens = [t.strip().lower() for t in query.split() if len(t) > 2]
        if not tokens:
            return None
        return {"tokens": tokens}
    
    try:
        # BM25 encoding - try query-specific method first
        if hasattr(_bm25, "encode_queries"):
            sparse = _bm25.encode_queries([query])[0]
        elif hasattr(_bm25, "encode_query"):
            sparse = _bm25.encode_query([query])[0]
        else:
            sparse = _bm25.encode_documents([query])[0]
        
        logger.debug("BM25 sparse encoding generated")
        return sparse
        
    except Exception as e:
        logger.warning(f"BM25 encoding failed: {e}")
        tokens = [t.strip().lower() for t in query.split() if len(t) > 2]
        return {"tokens": tokens} if tokens else None


async def _embed_query(query: str):
    """
    Legacy function - now uses Voyage for dense embedding.
    Maintained for backward compatibility.
    """
    return await _embed_query_voyage(query)


async def _query_dense_index(pinecone_client, index_name: str, query_vec, top_k: int = 50) -> List[Dict[str, Any]]:
    """Query a Pinecone dense index and return list of match dicts with metadata."""
    # index object depends on client wrapper
    try:
        index = pinecone_client.Index(index_name) if hasattr(pinecone_client, "Index") else pinecone_client.Index(index_name)
    except Exception:
        # some clients expose indexes differently
        index = pinecone_client.index(index_name)

    # run in thread if blocking
    def _q():
        try:
            return index.query(queries=[query_vec.tolist()], top_k=top_k, include_metadata=True)
        except TypeError:
            # older client signature
            return index.query(vector=query_vec.tolist(), top_k=top_k, include_metadata=True)

    resp = await asyncio.to_thread(_q)

    # Normalize response into a list of matches
    matches = []
    if isinstance(resp, dict):
        raw = resp.get("results") or resp.get("matches") or resp
    else:
        raw = getattr(resp, "matches", resp)

    # Support a variety of shapes
    entries = raw if isinstance(raw, list) else raw.get("matches", []) if isinstance(raw, dict) else []
    for m in entries:
        # Each match might already be a dict
        item = m if isinstance(m, dict) else (m.to_dict() if hasattr(m, "to_dict") else None)
        if item is None:
            continue
        matches.append({
            "id": item.get("id"),
            "score": item.get("score") or item.get("distance") or 0,
            "metadata": item.get("metadata") or item.get("fields") or {},
            "raw": item,
        })

    return matches


async def _query_sparse_index(pinecone_client, index_name: str, sparse_vector: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
    """Query a Pinecone sparse index. `sparse_vector` should be the representation expected by your Pinecone SDK."""
    try:
        index = pinecone_client.Index(index_name) if hasattr(pinecone_client, "Index") else pinecone_client.Index(index_name)
    except Exception:
        index = pinecone_client.index(index_name)

    def _q():
        # Try several common SDK signatures
        try:
            return index.query(sparse_vector=sparse_vector, top_k=top_k, include_metadata=True)
        except Exception:
            try:
                return index.query(queries=[{"sparse_vector": sparse_vector}], top_k=top_k, include_metadata=True)
            except Exception as e:
                raise

    resp = await asyncio.to_thread(_q)

    matches = []
    if isinstance(resp, dict):
        raw = resp.get("results") or resp.get("matches") or resp
    else:
        raw = getattr(resp, "matches", resp)

    entries = raw if isinstance(raw, list) else raw.get("matches", []) if isinstance(raw, dict) else []
    for m in entries:
        item = m if isinstance(m, dict) else (m.to_dict() if hasattr(m, "to_dict") else None)
        if item is None:
            continue
        matches.append({
            "id": item.get("id"),
            "score": item.get("score") or item.get("distance") or 0,
            "metadata": item.get("metadata") or item.get("fields") or {},
            "raw": item,
        })

    return matches


async def _rerank_with_anthropic(query: str, candidates: List[Dict[str, Any]], model: str = "claude-sonnet-4-20250514") -> List[Dict[str, Any]]:
    """Simple reranker that asks Anthropic to score each candidate for relevance.

    Returns candidates augmented with `rerank_score` float in [0,1].
    """
    try:
        from anthropic import Anthropic
    except Exception:
        raise RuntimeError("Anthropic client not available for reranking. Install and configure ANTHROPIC_API_KEY.")

    anth_key = os.getenv("ANTHROPIC_API_KEY")
    if not anth_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment for reranking")

    client = Anthropic(api_key=anth_key)

    # Build a compact prompt that enumerates candidates and asks for JSON scores
    parts = [f"Query: {query}", "\nCandidates:\n"]
    for i, c in enumerate(candidates, start=1):
        text = (c.get("metadata", {}).get("chunk_text") or c.get("metadata", {}).get("content") or c.get("metadata", {}).get("text") or "")
        parts.append(f"{i}. ID: {c.get('id')}\n{text[:1000]}\n")

    parts.append(
        "\nTask: For each candidate above, return a JSON array of objects [{\n  \"id\": <id>, \"score\": <0-1 float>\n}] where score is relevance to the query. Return ONLY valid JSON."
    )
    prompt = "\n".join(parts)

    # Call Anthropic in a thread
    def _call():
        return client.messages.create(model=model, max_tokens=800, temperature=0, messages=[{"role": "user", "content": prompt}])

    resp = await asyncio.to_thread(_call)
    raw = getattr(resp.content[0], "text", str(resp))

    # Extract JSON from response
    try:
        import re
        m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", raw)
        if m:
            arr = json.loads(m.group(1))
        else:
            arr = json.loads(raw)
    except Exception:
        logger.error("Failed to parse reranker response; raw output saved")
        logger.debug(raw[:2000])
        # Fallback: assign simple decreasing scores based on original score
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        for i, c in enumerate(sorted_candidates):
            c["rerank_score"] = max(0.0, 1.0 - (i * 0.05))
        return sorted_candidates

    # Map returned scores back to candidates
    id_to_score = {str(obj.get("id")): float(obj.get("score", 0)) for obj in arr}
    for c in candidates:
        c_id = str(c.get("id"))
        c["rerank_score"] = id_to_score.get(c_id, 0.0)

    return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)


async def _rerank_with_pinecone(pinecone_client, query: str, candidates: List[Dict[str, Any]], model: str = "pinecone-rerank-v0") -> List[Dict[str, Any]]:
    """Use Pinecone hosted reranker/model if available. Falls back to simple ordering on failure."""
    try:
        # Many Pinecone clients expose an `inference` helper
        def _call():
            try:
                return pinecone_client.inference.rerank(model=model, query=query, documents=[{"id": c.get("id"), "text": (c.get("metadata",{}).get("chunk_text") or c.get("metadata",{}).get("content") or "") } for c in candidates], top_n=len(candidates), return_documents=True)
            except Exception:
                # Alternate signature: use the single hybrid index name from env
                idx_name = os.getenv("PINECONE_INDEX_NAME")
                if not idx_name:
                    raise RuntimeError("PINECONE_INDEX_NAME not set for Pinecone rerank fallback")
                idx = pinecone_client.Index(idx_name)
                return idx.rerank(model=model, query=query, documents=[{"id": c.get("id"), "text": (c.get("metadata",{}).get("chunk_text") or c.get("metadata",{}).get("content") or "") } for c in candidates], top_n=len(candidates), return_documents=True)

        resp = await asyncio.to_thread(_call)
        # Parse resp for scores
        raw_matches = getattr(resp, "data", None) or resp.get("data", []) if isinstance(resp, dict) else []
        id_to_score = {}
        for item in raw_matches:
            cid = item.get("id") or (item.get("document", {}).get("id") if isinstance(item.get("document"), dict) else None)
            score = item.get("score") or item.get("relevance") or 0.0
            if cid:
                id_to_score[str(cid)] = float(score)

        for c in candidates:
            c["rerank_score"] = id_to_score.get(str(c.get("id")), 0.0)

        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception as e:
        logger.warning(f"Pinecone rerank failed: {e}")
        # fallback
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)


async def _rerank_with_cohere(query: str, candidates: List[Dict[str, Any]], model: str = "cohere-rerank-3.5") -> List[Dict[str, Any]]:
    """Use Cohere reranker if available; fall back to Anthropic ordering if not."""
    try:
        import cohere
    except Exception:
        logger.warning("Cohere client not available; falling back to Anthropic fallback rerank")
        return await _rerank_with_anthropic(query, candidates)

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        logger.warning("COHERE_API_KEY not set; falling back to Anthropic reranker")
        return await _rerank_with_anthropic(query, candidates)

    client = cohere.Client(COHERE_API_KEY)

    # Build documents list
    docs = [ (c.get("metadata",{}).get("chunk_text") or c.get("metadata",{}).get("content") or "") for c in candidates]
    try:
        resp = client.rerank(query=query, documents=docs, top_n=len(docs))
        scores = [0.0]*len(docs)
        for r in resp.ranked_results:
            idx = r.index
            scores[idx] = float(r.score)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = s
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception as e:
        logger.warning(f"Cohere rerank failed: {e}; falling back to Anthropic")
        return await _rerank_with_anthropic(query, candidates)


async def combined_search(
    query_text: str,
    index_name: str,
    top_k: int = 50,
    top_n: int = 10,
    rerank: bool = True,
    namespace: str = "research"
) -> List[Dict[str, Any]]:
    """
    HYBRID SEARCH: Combined dense + sparse retrieval with reranking.
    
    FLOW:
    1. Dense Embedding (Voyage-3-large) - captures meaning/semantics
    2. Sparse Embedding (BM25) - captures exact keywords
    3. Combined Query to Pinecone hybrid index
    4. Rerank candidates using configured provider (Anthropic/Cohere/Pinecone)
    5. Return top_n results
    
    Args:
        query_text: User query or document text
        index_name: Pinecone hybrid index name
        top_k: Number of initial candidates to retrieve
        top_n: Number of final results after reranking
        rerank: Whether to perform reranking
        namespace: Pinecone namespace (default: "research")
    
    Returns:
        List of candidate dicts with id, score, metadata, and rerank_score
    """
    logger.info(f"HYBRID SEARCH: '{query_text[:100]}...'")
    
    pc_client = _get_pinecone_client()

    # =========================================================
    # STEP 1: DENSE EMBEDDING (Voyage-3-large)
    # =========================================================
    # Captures: Intent, Context, Semantics
    logger.debug("Step 1: Generating dense embedding (Voyage-3-large)...")
    dense_vec = await _embed_query_voyage(query_text)
    
    if dense_vec is None:
        logger.error("Failed to generate dense embedding")
        return []
    
    # Ensure it's a list for Pinecone API
    if hasattr(dense_vec, 'tolist'):
        dense_vec = dense_vec.tolist()
    
    logger.debug(f"  → Dense vector: {len(dense_vec)} dimensions")

    # =========================================================
    # STEP 2: SPARSE EMBEDDING (BM25)
    # =========================================================
    # Captures: "NVIDIA", "share", "price" (exact keywords)
    logger.debug("Step 2: Generating sparse embedding (BM25)...")
    sparse_vec = _encode_sparse_bm25(query_text)
    logger.debug(f"  → Sparse vector generated: {sparse_vec is not None}")

    # =========================================================
    # STEP 3: HYBRID QUERY (Dense + Sparse combined)
    # =========================================================
    logger.debug(f"Step 3: Querying Pinecone hybrid index '{index_name}'...")
    
    try:
        index = pc_client.Index(index_name) if hasattr(pc_client, "Index") else pc_client.index(index_name)
    except Exception as e:
        logger.error(f"Failed to connect to index '{index_name}': {e}")
        return []

    def _execute_hybrid_query():
        """Execute hybrid query with multiple SDK signature fallbacks."""
        # Try modern Pinecone client signatures
        try:
            return index.query(
                vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
        except Exception:
            pass
        
        # Try older signature with queries list
        try:
            return index.query(
                queries=[{"values": dense_vec, "sparse_vector": sparse_vec}],
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
        except Exception:
            pass
        
        # Fallback to dense-only query
        try:
            logger.warning("Falling back to dense-only query")
            return index.query(
                vector=dense_vec,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
        except Exception as e:
            raise RuntimeError(f"All query methods failed: {e}")

    try:
        response = await asyncio.to_thread(_execute_hybrid_query)
    except Exception as e:
        logger.error(f"Hybrid query failed: {e}")
        return []

    # =========================================================
    # STEP 4: PARSE RESULTS
    # =========================================================
    matches = []
    raw_matches = getattr(response, "matches", None) or response.get("matches", [])
    
    for m in raw_matches:
        if isinstance(m, dict):
            item = m
        elif hasattr(m, "to_dict"):
            item = m.to_dict()
        else:
            continue
        
        matches.append({
            "id": item.get("id"),
            "score": item.get("score") or item.get("distance") or 0,
            "metadata": item.get("metadata") or item.get("fields") or {},
        })

    logger.info(f"  → Retrieved {len(matches)} candidates from hybrid search")

    if not matches:
        return []

    # =========================================================
    # STEP 5: RERANK (Optional but recommended)
    # =========================================================
    if rerank and matches:
        logger.debug(f"Step 4: Reranking with provider '{RERANKER_PROVIDER}'...")
        
        if RERANKER_PROVIDER == "pinecone":
            reranked = await _rerank_with_pinecone(pc_client, query_text, matches)
        elif RERANKER_PROVIDER == "cohere":
            reranked = await _rerank_with_cohere(query_text, matches)
        else:
            # Default: Anthropic reranker
            reranked = await _rerank_with_anthropic(query_text, matches)
        
        logger.info(f"  → Returning top {min(top_n, len(reranked))} after reranking")
        return reranked[:top_n]

    # No reranking - sort by original score
    sorted_candidates = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_candidates[:top_n]


async def rerank_candidates(query_text: str, candidates: List[Dict[str, Any]], provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Public helper to rerank an existing candidate list using the configured or specified provider.

    Returns candidates augmented with `rerank_score` and sorted descending.
    """
    pc_client = _get_pinecone_client()
    provider = (provider or RERANKER_PROVIDER).lower()
    if provider == "pinecone":
        return await _rerank_with_pinecone(pc_client, query_text, candidates)
    elif provider == "cohere":
        return await _rerank_with_cohere(query_text, candidates)
    else:
        return await _rerank_with_anthropic(query_text, candidates)
