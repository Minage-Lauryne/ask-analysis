"""
hybrid_retrieval.py

Complete Hybrid RAG Pipeline for Single Analysis:
    User Question/Document
           ↓
    [If Document: Chunk with 500 tokens, 50 overlap]
           ↓
    Voyage Embedding (dense/meaning) + BM25 (sparse/keywords)
           ↓
    Dense Search      Sparse Search
           ↓              ↓
           └── Combine Results ──┘
                   ↓
               Reranker
                   ↓
              Top Documents
                   ↓
                  LLM
                   ↓
              Final Answer

This module handles:
1. Input classification (document vs question)
2. Document chunking (500 tokens, 50 overlap)
3. Dense embedding (Voyage-3-large)
4. Sparse embedding (BM25/Pinecone)
5. Hybrid search (dense + sparse)
6. Result combination
7. Reranking
8. Citation formatting with REF ID [1], [2], etc.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATION
# =========================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-hybrid")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Chunking parameters
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4  # Approximation: 1 token ≈ 4 characters

# =========================================================
# CLIENT INITIALIZATION
# =========================================================

# Voyage client for dense embeddings
voyage_client = None
try:
    import voyageai
    if VOYAGE_API_KEY:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        logger.info("Voyage client initialized for dense embeddings")
    else:
        logger.warning("VOYAGE_API_KEY not set - dense embeddings unavailable")
except ImportError:
    logger.warning("voyageai package not installed")

# BM25 encoder for sparse embeddings
bm25_encoder = None
try:
    from pinecone_text.sparse import BM25Encoder
    bm25_encoder = BM25Encoder().default()
    logger.info("BM25 encoder initialized for sparse embeddings")
except ImportError:
    logger.warning("pinecone_text package not installed - sparse embeddings unavailable")
except Exception as e:
    logger.warning(f"BM25 encoder initialization failed: {e}")

# Pinecone client
pinecone_client = None
pinecone_index = None
try:
    from pinecone import Pinecone
    if PINECONE_API_KEY:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME:
            pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' connected")
except ImportError:
    logger.warning("pinecone package not installed")
except Exception as e:
    logger.warning(f"Pinecone initialization failed: {e}")

# Anthropic client for LLM
anthropic_client = None
try:
    from anthropic import Anthropic
    if ANTHROPIC_API_KEY:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized for LLM")
except ImportError:
    logger.warning("anthropic package not installed")


# =========================================================
# STEP 1: INPUT CLASSIFICATION & CHUNKING
# =========================================================

def is_document(text: str) -> bool:
    """
    Determine if the input is a document (requires chunking) or a question.
    
    Documents typically have:
    - Length > 2000 characters
    - Multiple paragraphs
    - High word count
    
    Questions are typically:
    - Short
    - Single line or paragraph
    - Interrogative
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Length-based heuristics
    if len(text) > 2000:
        return True
    
    # Multiple paragraphs
    if text.count("\n\n") > 2:
        return True
    
    # High word count
    word_count = len(text.split())
    if word_count > 400:
        return True
    
    # Multiple sentences (likely a document)
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    if sentence_count > 10:
        return True
    
    return False


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    prefix: str = "chunk"
) -> List[Dict[str, Any]]:
    """
    Chunk text into fixed-size chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_tokens: Maximum tokens per chunk (default 500)
        overlap_tokens: Tokens to overlap between chunks (default 50)
        prefix: Prefix for chunk IDs
    
    Returns:
        List of chunk dictionaries with:
        - chunk_id: Unique identifier
        - content: Chunk text
        - token_count: Estimated token count
        - start_char: Start position in original text
        - end_char: End position in original text
    """
    if not text:
        return []
    
    text = text.strip()
    
    # Convert tokens to characters
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    step_size = max(1, max_chars - overlap_chars)
    
    chunks = []
    start = 0
    chunk_num = 1
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text_content = text[start:end].strip()
        
        # Skip very small chunks
        if len(chunk_text_content) < 50:
            break
        
        chunks.append({
            "chunk_id": f"{prefix}_{chunk_num}",
            "content": chunk_text_content,
            "token_count": len(chunk_text_content) // CHARS_PER_TOKEN,
            "start_char": start,
            "end_char": end,
            "chunk_number": chunk_num
        })
        
        chunk_num += 1
        start += step_size
    
    logger.info(f"Created {len(chunks)} chunks from text ({len(text)} chars)")
    return chunks


async def llm_chunk_document(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """
    Use LLM (Anthropic) to create semantic chunks with proper overlap.
    Falls back to fixed-size chunking if LLM is unavailable.
    
    Args:
        text: Document text to chunk
        chunk_tokens: Target tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of chunk dictionaries
    """
    # Try LLM-based chunking first
    if anthropic_client:
        try:
            prompt = f"""You are a document chunker. Split the provided document into semantic chunks.

Rules:
1. Each chunk should be approximately {chunk_tokens} tokens (~{chunk_tokens * CHARS_PER_TOKEN} characters)
2. OVERLAP REQUIREMENT: Start each chunk (after the first) with the LAST {overlap_tokens} tokens from the previous chunk
3. Break at natural topic boundaries (paragraphs, sections, ideas)
4. Fix any OCR errors or formatting issues
5. Generate a brief topic description for each chunk

Return ONLY valid JSON array:
[
  {{
    "chunk_number": 1,
    "content": "chunk text here...",
    "topic": "brief topic description",
    "token_count": estimated_token_count
  }}
]

Document to chunk:
{text[:15000]}"""  # Limit to avoid token limits

            def _call_llm():
                return anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            response = await asyncio.to_thread(_call_llm)
            raw_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                if len(parts) >= 2:
                    raw_text = parts[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()
            
            chunks_data = json.loads(raw_text)
            
            # Normalize to our format
            chunks = []
            for i, item in enumerate(chunks_data, start=1):
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                chunks.append({
                    "chunk_id": f"doc_chunk_{i}",
                    "content": content,
                    "topic": item.get("topic", "") if isinstance(item, dict) else "",
                    "token_count": len(content) // CHARS_PER_TOKEN,
                    "chunk_number": i
                })
            
            if chunks:
                logger.info(f"LLM created {len(chunks)} semantic chunks")
                return chunks
                
        except Exception as e:
            logger.warning(f"LLM chunking failed: {e}, falling back to fixed-size")
    
    # Fallback to fixed-size chunking
    return chunk_text(text, chunk_tokens, overlap_tokens, prefix="doc_chunk")


# =========================================================
# STEP 2: EMBEDDINGS (Dense + Sparse)
# =========================================================

async def generate_dense_embedding(
    text: str,
    model: str = "voyage-3-large",
    input_type: str = "query"
) -> Optional[List[float]]:
    """
    Generate dense embedding using Voyage AI.
    
    Args:
        text: Text to embed
        model: Voyage model (voyage-3-large recommended)
        input_type: "query" for search queries, "document" for documents
    
    Returns:
        Dense embedding vector or None if failed
    """
    if not voyage_client:
        logger.error("Voyage client not available for dense embedding")
        return None
    
    try:
        def _embed():
            return voyage_client.embed(
                texts=[text],
                model=model,
                input_type=input_type
            )
        
        response = await asyncio.to_thread(_embed)
        
        # Extract embedding from response
        embeddings = getattr(response, "embeddings", None)
        if embeddings and isinstance(embeddings, list) and embeddings:
            logger.debug(f"Dense embedding generated: {len(embeddings[0])} dimensions")
            return embeddings[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Dense embedding failed: {e}")
        return None


def generate_sparse_embedding(text: str) -> Optional[Dict[str, Any]]:
    """
    Generate sparse embedding using BM25.
    
    Args:
        text: Text to encode
    
    Returns:
        Sparse vector dict or None if failed
    """
    if not bm25_encoder:
        logger.warning("BM25 encoder not available for sparse embedding")
        # Fallback to simple token-based sparse vector
        tokens = [t.strip().lower() for t in text.split() if len(t) > 2]
        return {"tokens": tokens} if tokens else None
    
    try:
        # BM25 encoding for queries
        if hasattr(bm25_encoder, "encode_queries"):
            sparse = bm25_encoder.encode_queries([text])[0]
        elif hasattr(bm25_encoder, "encode_query"):
            sparse = bm25_encoder.encode_query([text])[0]
        else:
            sparse = bm25_encoder.encode_documents([text])[0]
        
        logger.debug("Sparse embedding generated via BM25")
        return sparse
        
    except Exception as e:
        logger.warning(f"BM25 encoding failed: {e}, using fallback")
        tokens = [t.strip().lower() for t in text.split() if len(t) > 2]
        return {"tokens": tokens} if tokens else None


async def generate_hybrid_embeddings(
    text: str,
    input_type: str = "query"
) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]]]:
    """
    Generate both dense and sparse embeddings for a text.
    
    Args:
        text: Text to embed
        input_type: "query" or "document"
    
    Returns:
        Tuple of (dense_embedding, sparse_embedding)
    """
    # Generate both embeddings
    dense_embedding = await generate_dense_embedding(text, input_type=input_type)
    sparse_embedding = generate_sparse_embedding(text)
    
    return dense_embedding, sparse_embedding


# =========================================================
# STEP 3: HYBRID SEARCH (Dense + Sparse)
# =========================================================

async def hybrid_search(
    query_text: str,
    index_name: Optional[str] = None,
    top_k: int = 50,
    namespace: str = "research"
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining dense and sparse retrieval.
    
    Args:
        query_text: Query text
        index_name: Pinecone index name (defaults to PINECONE_INDEX_NAME)
        top_k: Number of results to retrieve
        namespace: Pinecone namespace
    
    Returns:
        List of matches with id, score, metadata
    """
    index_name = index_name or PINECONE_INDEX_NAME
    
    if not pinecone_client:
        logger.error("Pinecone client not available for search")
        return []
    
    try:
        index = pinecone_client.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
        return []
    
    # Generate hybrid embeddings for query
    dense_vec, sparse_vec = await generate_hybrid_embeddings(query_text, input_type="query")
    
    if not dense_vec:
        logger.error("Failed to generate dense embedding for query")
        return []
    
    # Execute hybrid query
    try:
        def _query():
            # Try different query signatures for compatibility
            try:
                # Modern Pinecone client with hybrid support
                return index.query(
                    vector=dense_vec,
                    sparse_vector=sparse_vec,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
            except Exception:
                # Fallback to dense-only query
                return index.query(
                    vector=dense_vec,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )
        
        response = await asyncio.to_thread(_query)
        
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return []
    
    # Parse response into standard format
    matches = []
    raw_matches = getattr(response, "matches", []) or response.get("matches", [])
    
    for m in raw_matches:
        if isinstance(m, dict):
            item = m
        elif hasattr(m, "to_dict"):
            item = m.to_dict()
        else:
            continue
        
        matches.append({
            "id": item.get("id"),
            "score": item.get("score", 0),
            "metadata": item.get("metadata") or item.get("fields") or {},
        })
    
    logger.info(f"Hybrid search returned {len(matches)} results")
    return matches


# =========================================================
# STEP 4: COMBINE & DEDUPLICATE RESULTS
# =========================================================

def combine_search_results(
    results_list: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Combine multiple search result lists, deduplicating by ID
    and keeping the highest score for each document.
    
    Args:
        results_list: List of result lists to combine
    
    Returns:
        Deduplicated list of results
    """
    candidate_map: Dict[str, Dict[str, Any]] = {}
    
    for results in results_list:
        for result in results:
            doc_id = str(result.get("id", ""))
            if not doc_id:
                continue
            
            # Keep result with highest score
            if doc_id not in candidate_map:
                candidate_map[doc_id] = result
            elif result.get("score", 0) > candidate_map[doc_id].get("score", 0):
                candidate_map[doc_id] = result
    
    combined = list(candidate_map.values())
    logger.info(f"Combined {sum(len(r) for r in results_list)} results into {len(combined)} unique candidates")
    
    return combined


# =========================================================
# STEP 5: RERANKING
# =========================================================

async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 10,
    model: str = "claude-3-5-sonnet-20241022"
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Anthropic LLM to score relevance.
    
    Args:
        query: Original query
        candidates: List of candidate results
        top_n: Number of top results to return
        model: Anthropic model for reranking
    
    Returns:
        Reranked list of candidates with rerank_score
    """
    if not candidates:
        return []
    
    if not anthropic_client:
        logger.warning("Anthropic client not available for reranking, using original scores")
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_n]
    
    # Build prompt for reranking
    prompt_parts = [f"Query: {query}\n\nCandidates:\n"]
    
    for i, c in enumerate(candidates[:50], start=1):  # Limit to 50 for API constraints
        md = c.get("metadata", {})
        text = (
            md.get("chunk_text") or 
            md.get("content") or 
            md.get("text") or 
            md.get("citation_text") or 
            ""
        )[:1000]  # Truncate for API limits
        
        title = md.get("study_title") or md.get("filename") or "Untitled"
        prompt_parts.append(f"{i}. ID: {c.get('id')}\nTitle: {title}\nContent: {text}\n")
    
    prompt_parts.append(
        "\nTask: Score each candidate's relevance to the query (0.0 to 1.0).\n"
        "Return ONLY a valid JSON array: [{\"id\": \"...\", \"score\": 0.85}, ...]\n"
        "Higher scores = more relevant."
    )
    
    prompt = "\n".join(prompt_parts)
    
    try:
        def _call():
            return anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
        
        response = await asyncio.to_thread(_call)
        raw = response.content[0].text.strip()
        
        # Parse JSON scores
        import re
        match = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", raw)
        if match:
            scores_data = json.loads(match.group(1))
        else:
            scores_data = json.loads(raw)
        
        # Map scores back to candidates
        id_to_score = {str(s.get("id")): float(s.get("score", 0)) for s in scores_data}
        
        for c in candidates:
            c_id = str(c.get("id"))
            c["rerank_score"] = id_to_score.get(c_id, 0.0)
        
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(f"Reranked {len(candidates)} candidates, returning top {top_n}")
        return reranked[:top_n]
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback to original scores
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_n]


# =========================================================
# STEP 6: CITATION FORMATTING (REF ID System)
# =========================================================

def format_apa_citation(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into APA-style citation.
    
    Args:
        metadata: Document metadata
    
    Returns:
        Formatted APA citation string
    """
    # Check for explicit full citation first
    full_citation = (
        metadata.get("full_citation") or 
        metadata.get("Full Citation") or
        metadata.get("citation")
    )
    if full_citation:
        return full_citation
    
    # Build APA citation from components
    parts = []
    
    # Authors
    authors = (
        metadata.get("authors") or 
        metadata.get("author") or 
        metadata.get("Author") or
        ""
    )
    if authors:
        parts.append(str(authors))
    
    # Year
    year = (
        metadata.get("year") or 
        metadata.get("Year") or 
        metadata.get("Year ") or  # Handle trailing space
        "n.d."
    )
    parts.append(f"({year})")
    
    # Title
    title = (
        metadata.get("study_title") or 
        metadata.get("Study Title") or
        metadata.get("title") or
        metadata.get("filename") or
        ""
    )
    if title:
        # Clean up filename-style titles
        title = title.replace(".pdf", "").replace("_", " ").strip()
        parts.append(title)
    
    # Source/Journal
    source = metadata.get("source") or metadata.get("journal") or ""
    if source:
        parts.append(source)
    
    return ". ".join([p for p in parts if p]) + "."


def build_numbered_context(
    matches: List[Dict[str, Any]],
    max_content_chars: int = 2000
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build numbered research context with REF ID citations.
    
    Args:
        matches: List of search results
        max_content_chars: Maximum characters per content excerpt
    
    Returns:
        Tuple of (formatted_context, citations_list)
    """
    context_parts = []
    citations = []
    
    for idx, m in enumerate(matches, start=1):
        md = m.get("metadata", {})
        
        # Extract content
        content = (
            md.get("chunk_text") or 
            md.get("content") or 
            md.get("text") or
            md.get("citation_text") or
            ""
        )[:max_content_chars]
        
        # Extract title
        title = (
            md.get("study_title") or 
            md.get("Study Title") or
            md.get("filename") or 
            "Untitled Document"
        )
        
        # Format citation
        citation = format_apa_citation(md)
        
        # Get link
        link = (
            md.get("study_link") or 
            md.get("gdrive_link") or
            md.get("Link to Full Study") or
            md.get("link") or
            ""
        )
        
        # Build context block
        block = (
            f"### REF ID [{idx}]\n"
            f"TITLE: {title}\n"
            f"FULL CITATION: {citation}\n"
            f"LINK: {link}\n"
            f"CONTENT EXCERPTS:\n{content}\n"
        )
        context_parts.append(block)
        
        # Store citation metadata
        citations.append({
            "ref_id": idx,
            "id": m.get("id"),
            "title": title,
            "full_citation": citation,
            "link": link,
            "score": m.get("rerank_score") or m.get("score", 0),
            "content_preview": content[:300],
            "metadata": md
        })
    
    context = "\n\n".join(context_parts)
    logger.info(f"Built numbered context with {len(citations)} REF IDs")
    
    return context, citations


# =========================================================
# STEP 7: LLM GENERATION
# =========================================================

async def generate_analysis(
    query: str,
    context: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """
    Generate final analysis using LLM with research context.
    
    Args:
        query: User query
        context: Numbered research context
        system_prompt: Custom system prompt (optional)
        max_tokens: Maximum response tokens
        model: Anthropic model
    
    Returns:
        Dict with response and metadata
    """
    if not anthropic_client:
        return {
            "response": "Error: Anthropic client not available for generation.",
            "error": True
        }
    
    default_system = """You are a Research Analyst AI.
Use the provided RESEARCH CONTEXT to answer the USER QUERY.

CRITICAL INSTRUCTIONS:
1. **Data Density:** Extract specific statistics, percentages, ages, and sample sizes from the text. Do not generalize if specific numbers are available.
2. **Inline Citations:** The context provides studies labeled as REF ID [1], [2], etc.
   - When you state a fact, IMMEDIATELY reference the study ID in brackets: "Educational engagement was high (95%) [1]."
3. **Citation Matching:** Reference the REF IDs in your response when using information from those sources.

Format your response with clear sections and include inline citations [1], [2], etc."""

    final_system = system_prompt or default_system
    
    user_message = f"""USER QUERY: "{query}"

RESEARCH CONTEXT:
{context}"""

    try:
        def _call():
            return anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=final_system,
                messages=[{"role": "user", "content": user_message}]
            )
        
        response = await asyncio.to_thread(_call)
        raw_text = response.content[0].text.strip()
        
        return {
            "response": raw_text,
            "model": model,
            "error": False
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return {
            "response": f"Error generating analysis: {str(e)}",
            "error": True
        }


# =========================================================
# MAIN PIPELINE: COMPLETE HYBRID RAG FLOW
# =========================================================

async def hybrid_rag_pipeline(
    query_text: str,
    index_name: Optional[str] = None,
    top_k_retrieval: int = 50,
    top_n_rerank: int = 10,
    chunk_documents: bool = True,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4000,
    namespace: str = "research"
) -> Dict[str, Any]:
    """
    Complete Hybrid RAG Pipeline for Single Analysis.
    
    Flow:
    1. Input Classification: Document vs Question
    2. If Document: Chunk (500 tokens, 50 overlap)
    3. Generate Embeddings: Dense (Voyage) + Sparse (BM25)
    4. Hybrid Search: Dense + Sparse combined
    5. Combine & Deduplicate results
    6. Rerank candidates
    7. Build numbered context with REF ID citations
    8. Generate analysis with LLM
    
    Args:
        query_text: User query or document
        index_name: Pinecone index name
        top_k_retrieval: Number of results per search
        top_n_rerank: Number of results after reranking
        chunk_documents: Whether to chunk document inputs
        system_prompt: Custom system prompt for LLM
        max_tokens: Max tokens for LLM response
        namespace: Pinecone namespace
    
    Returns:
        Dict containing:
        - response: Generated analysis
        - citations: List of citation metadata
        - context: Numbered research context
        - chunks: List of query chunks (if document)
        - candidates: All candidate results before reranking
        - top_results: Top results after reranking
    """
    logger.info("=" * 60)
    logger.info("HYBRID RAG PIPELINE - SINGLE ANALYSIS")
    logger.info("=" * 60)
    
    index_name = index_name or PINECONE_INDEX_NAME
    
    result = {
        "response": "",
        "citations": [],
        "context": "",
        "chunks": [],
        "candidates": [],
        "top_results": [],
        "source": "hybrid_rag",
        "error": False
    }
    
    # STEP 1: Classify input
    is_doc = is_document(query_text) if chunk_documents else False
    logger.info(f"Step 1: Input classified as {'DOCUMENT' if is_doc else 'QUESTION'}")
    
    # STEP 2: Chunk if document
    if is_doc:
        logger.info("Step 2: Chunking document (500 tokens, 50 overlap)...")
        chunks = await llm_chunk_document(query_text)
        result["chunks"] = chunks
        logger.info(f"  → Created {len(chunks)} chunks")
    else:
        logger.info("Step 2: Skipping chunking (input is a question)")
        chunks = [{"chunk_id": "query", "content": query_text}]
    
    # STEP 3 & 4: Search for each chunk/query
    logger.info("Steps 3-4: Hybrid search (Dense + Sparse)...")
    all_results = []
    
    for chunk in chunks:
        chunk_query = chunk.get("content", "")
        if not chunk_query.strip():
            continue
        
        # Hybrid search for this chunk
        matches = await hybrid_search(
            query_text=chunk_query,
            index_name=index_name,
            top_k=top_k_retrieval,
            namespace=namespace
        )
        all_results.append(matches)
        logger.info(f"  → Chunk '{chunk.get('chunk_id')}': {len(matches)} results")
    
    # STEP 5: Combine and deduplicate
    logger.info("Step 5: Combining and deduplicating results...")
    candidates = combine_search_results(all_results)
    result["candidates"] = candidates
    logger.info(f"  → {len(candidates)} unique candidates")
    
    if not candidates:
        logger.warning("No candidates found, returning empty result")
        result["response"] = "No relevant research found for this query."
        return result
    
    # STEP 6: Rerank
    logger.info("Step 6: Reranking candidates...")
    # Use original query for reranking, not chunks
    top_results = await rerank_results(
        query=query_text[:2000],  # Truncate for reranker
        candidates=candidates,
        top_n=top_n_rerank
    )
    result["top_results"] = top_results
    logger.info(f"  → Top {len(top_results)} results selected")
    
    # STEP 7: Build numbered context
    logger.info("Step 7: Building numbered context with REF IDs...")
    context, citations = build_numbered_context(top_results)
    result["context"] = context
    result["citations"] = citations
    
    # STEP 8: Generate analysis
    logger.info("Step 8: Generating analysis with LLM...")
    generation_result = await generate_analysis(
        query=query_text,
        context=context,
        system_prompt=system_prompt,
        max_tokens=max_tokens
    )
    
    result["response"] = generation_result.get("response", "")
    result["error"] = generation_result.get("error", False)
    
    logger.info("=" * 60)
    logger.info("HYBRID RAG PIPELINE COMPLETE")
    logger.info(f"  → Response length: {len(result['response'])} chars")
    logger.info(f"  → Citations: {len(citations)}")
    logger.info("=" * 60)
    
    return result


# =========================================================
# RETRIEVAL-ONLY FUNCTION (No LLM Generation)
# =========================================================

async def hybrid_retrieval_only(
    query_text: str,
    index_name: Optional[str] = None,
    top_k_retrieval: int = 50,
    top_n_rerank: int = 10,
    chunk_documents: bool = True,
    namespace: str = "research"
) -> Dict[str, Any]:
    """
    Perform hybrid retrieval without LLM generation.
    
    Useful when you want to integrate with existing generation logic.
    
    Returns:
        Dict containing:
        - top_results: Reranked results
        - citations: Citation metadata
        - context: Numbered context string
        - chunks: Query chunks (if document)
        - candidates: All candidates before reranking
    """
    index_name = index_name or PINECONE_INDEX_NAME
    
    result = {
        "top_results": [],
        "citations": [],
        "context": "",
        "chunks": [],
        "candidates": []
    }
    
    # Classify and chunk
    is_doc = is_document(query_text) if chunk_documents else False
    
    if is_doc:
        chunks = await llm_chunk_document(query_text)
        result["chunks"] = chunks
    else:
        chunks = [{"chunk_id": "query", "content": query_text}]
    
    # Search for each chunk
    all_results = []
    for chunk in chunks:
        chunk_query = chunk.get("content", "")
        if not chunk_query.strip():
            continue
        
        matches = await hybrid_search(
            query_text=chunk_query,
            index_name=index_name,
            top_k=top_k_retrieval,
            namespace=namespace
        )
        all_results.append(matches)
    
    # Combine and rerank
    candidates = combine_search_results(all_results)
    result["candidates"] = candidates
    
    if not candidates:
        return result
    
    top_results = await rerank_results(
        query=query_text[:2000],
        candidates=candidates,
        top_n=top_n_rerank
    )
    result["top_results"] = top_results
    
    # Build context
    context, citations = build_numbered_context(top_results)
    result["context"] = context
    result["citations"] = citations
    
    return result
