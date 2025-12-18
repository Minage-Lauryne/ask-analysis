"""
pinecone_rag.py

Provides a high-level RAG helper that performs:
 - Dense embedding (via app.services.embeddings.embed_text)
 - Dense index query (Pinecone)
 - Sparse index query (Pinecone sparse index) if available
 - Candidate merging (union of top-K)
 - Reranking via Anthropic LLM (simple JSON scorer)

This file is intentionally defensive: if Pinecone or Anthropic is not
installed/configured, it raises informative errors and documents next steps.

Usage:
    from app.services.pinecone_rag import combined_search
    # Use a single hybrid Pinecone index (dense `values` + `sparse_values`)
    results = await combined_search(query_text, index_name="my-hybrid-index")

"""
from __future__ import annotations

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

from app.services.embeddings import embed_text

logger = logging.getLogger(__name__)

# Optional BM25 sparse encoder (used for query encoding if available)
try:
    from pinecone_text.sparse import BM25Encoder
    _BM25_AVAILABLE = True
except Exception:
    BM25Encoder = None
    _BM25_AVAILABLE = False

# instantiate BM25 encoder if available
_bm25 = None
if _BM25_AVAILABLE:
    try:
        _bm25 = BM25Encoder().default()
    except Exception:
        _bm25 = None

# RERANKER_PROVIDER: 'anthropic' | 'pinecone' | 'cohere'
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "anthropic").lower()


def _get_pinecone_client():
    try:
        # Prefer the lightweight Pinecone client wrapper if present
        try:
            from pinecone import Pinecone as PineconeClient
            return PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        except Exception:
            import pinecone as pc
            # initialize if not already
            api_key = os.getenv("PINECONE_API_KEY")
            # Prefer new `PINECONE_INDEX_HOST`; avoid legacy `PINECONE_ENV` usage
            env = os.getenv("PINECONE_INDEX_HOST")
            if api_key:
                if env:
                    pc.init(api_key=api_key, environment=env)
                else:
                    pc.init(api_key=api_key)
            return pc
    except Exception as e:
        raise RuntimeError("Pinecone client not available or not configured: " + str(e))


async def _embed_query(query: str):
    # embed_text is synchronous; run in thread to avoid blocking
    return await asyncio.to_thread(embed_text, query)


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
) -> List[Dict[str, Any]]:
    """
    Perform combined dense + sparse retrieval and rerank.

    Returns top_n candidate dicts containing id, metadata and rerank_score.
    """
    pc_client = _get_pinecone_client()

    # 1) dense embed the query
    q_vec = await _embed_query(query_text)

    # 2) prepare sparse query vector (BM25Encoder preferred)
    sparse_vector = None
    if _bm25 is not None:
        try:
            if hasattr(_bm25, "encode_query"):
                sparse_vector = _bm25.encode_query([query_text])[0]
            else:
                sparse_vector = _bm25.encode_documents([query_text])[0]
        except Exception as e:
            logger.debug(f"BM25 encoding failed: {e}")

    if sparse_vector is None:
        tokens = [t.strip().lower() for t in query_text.split() if len(t) > 2]
        sparse_vector = {"tokens": tokens}

    # 3) query single hybrid index (try several SDK signatures)
    try:
        try:
            index = pc_client.Index(index_name) if hasattr(pc_client, "Index") else pc_client.index(index_name)
        except Exception:
            index = pc_client.index(index_name)

        def _q_hybrid():
            # Try modern signatures first
            try:
                return index.query(queries=[{"values": q_vec.tolist(), "sparse_vector": sparse_vector}], top_k=top_k, include_metadata=True)
            except Exception:
                pass
            try:
                return index.query(vector=q_vec.tolist(), sparse_vector=sparse_vector, top_k=top_k, include_metadata=True)
            except Exception:
                pass
            try:
                # older clients: pass sparse_vector inside queries
                return index.query(queries=[{"vector": q_vec.tolist(), "sparse_vector": sparse_vector}], top_k=top_k, include_metadata=True)
            except Exception as e:
                raise

        resp = await asyncio.to_thread(_q_hybrid)

    except Exception as e:
        logger.warning(f"Hybrid query failed for index {index_name}: {e}")
        return []

    # Normalize response into matches list
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

    # candidates are simply the matches (single-source)
    candidates = matches

    if not candidates:
        return []
    # 5) optionally rerank candidates using configured provider
    if rerank:
        provider = RERANKER_PROVIDER
        if provider == "pinecone":
            reranked = await _rerank_with_pinecone(pc_client, query_text, candidates)
        elif provider == "cohere":
            reranked = await _rerank_with_cohere(query_text, candidates)
        else:
            # default: anthropic
            reranked = await _rerank_with_anthropic(query_text, candidates)

        return reranked[:top_n]

    # If rerank is disabled, sort by original score descending and return top_n
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
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
