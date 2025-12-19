import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from app.services.embeddings import embed_text
from app.services.pinecone_rag import combined_search, rerank_candidates
import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_DB_URL = os.getenv("DB_URL")

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the global connection pool."""
    global _pool
    if _pool is None or _pool._closed:
        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL (or DATABASE_URL) must be set")
        _pool = await asyncpg.create_pool(
            dsn=SUPABASE_DB_URL, 
            min_size=1, 
            max_size=10, 
            command_timeout=30,
            statement_cache_size=0
        )
    return _pool


async def close_pool():
    """Close the global connection pool (call on app shutdown)."""
    global _pool
    if _pool is not None and not _pool._closed:
        await _pool.close()
        _pool = None


async def search_research_chunks_from_text(
    query_text: str,
    top_k: int = 10,
    domain: Optional[str] = None,
    include_pdf_urls: bool = True
) -> List[Dict[str, Any]]:
    """
    1. Embed query_text using your existing embed_text()
    2. Call agent.search_research_chunks on Supabase Postgres
    3. Return structured research items for use in prompts
    """

    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL (or DATABASE_URL) must be set")

    logger.debug(f"Generating embedding for query: '{query_text[:100]}...'")
    query_embedding = embed_text(query_text)
    if query_embedding is None:
        raise ValueError("Failed to generate embedding for query_text")

    # Prefer Pinecone RAG flow when `PINECONE_INDEX_NAME` is set (single hybrid index)
    index_name = os.getenv("PINECONE_INDEX_NAME")

    def _is_document(text: str) -> bool:
        # Heuristic: long text or multiple paragraphs -> treat as document
        if not text:
            return False
        if len(text) > 2000:
            return True
        if "\n\n" in text:
            return True
        if text.strip().count(" ") > 400:
            return True
        return False

    async def _llm_chunk_document(text: str) -> List[Dict[str, Any]]:
        """Chunk a long document into ~500-token chunks with 50-token overlap using Anthropic.

        Falls back to a deterministic fixed-size chunker if Anthropic is not available.
        Returns list of chunk dicts with keys: chunk_id, content, token_count
        """
        # simple character-based fallback chunker
        def _fallback_chunks(t: str):
            # 1 token ~ 4 chars approximation
            max_chars = 500 * 4
            overlap_chars = 50 * 4
            step = max_chars - overlap_chars
            chunks = []
            i = 0
            counter = 1
            while i < len(t):
                part = t[i:i + max_chars].strip()
                if not part:
                    break
                chunks.append({
                    "chunk_id": f"doc_chunk_{counter}",
                    "content": part,
                    "token_count": len(part) // 4
                })
                counter += 1
                i += step
            return chunks

        # Try Anthropic first
        try:
            from anthropic import Anthropic
            anth_key = os.getenv("ANTHROPIC_API_KEY")
            if anth_key:
                client = Anthropic(api_key=anth_key)
                prompt = (
                    "You are a document chunker. Split the provided document into semantic chunks.\n"
                    "Rules:\n"
                    "- Each chunk should be ~500 tokens (approx).\n"
                    "- Each chunk after the first should start with the last 50 tokens of the previous chunk (overlap).\n"
                    "- Return ONLY valid JSON array of objects: [{\n  'chunk_number': 1, 'content': '...', 'token_count': 123\n}]"
                )
                # send the whole text as the user content (may be long)
                def _call():
                    return client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=8000,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt + "\n\n" + text}]
                    )

                resp = await asyncio.to_thread(_call)
                raw = getattr(resp.content[0], "text", "")
                # strip markdown fences
                if raw.startswith("```"):
                    parts = raw.split("```")
                    if len(parts) >= 2:
                        raw = parts[1].strip()
                try:
                    arr = json.loads(raw)
                    chunks = []
                    for i, item in enumerate(arr, start=1):
                        content = item.get("content") if isinstance(item, dict) else str(item)
                        chunks.append({
                            "chunk_id": f"doc_chunk_{i}",
                            "content": content,
                            "token_count": len(content) // 4
                        })
                    if chunks:
                        return chunks
                except Exception:
                    # fall through to fallback
                    pass
        except Exception:
            pass

        return _fallback_chunks(text)

    # If Pinecone index is configured, use combined_search RAG path
    if index_name:
        try:
            results: List[Dict[str, Any]] = []
            if _is_document(query_text):
                # Chunk document and search per-chunk then aggregate
                chunks = await _llm_chunk_document(query_text)
                logger.info(f"Document query split into {len(chunks)} chunks")

                candidate_map: Dict[str, Dict[str, Any]] = {}
                for ch in chunks:
                    q = ch.get("content", "")
                    if not q.strip():
                        continue
                    matches = await combined_search(q, index_name=index_name, top_k=20, top_n=50, rerank=False)
                    for m in matches:
                        mid = m.get("id")
                        if mid not in candidate_map:
                            candidate_map[mid] = m

                if not candidate_map:
                    return []

                candidates = list(candidate_map.values())

                # Global rerank against the original query_text
                reranked = await rerank_candidates(query_text, candidates)
                top = reranked[:top_k]

                # Map to expected output shape
                for m in top:
                    md = m.get("metadata", {})
                    results.append({
                        "id": m.get("id"),
                        "chunk_id": md.get("chunk_id") or md.get("chunk_id", m.get("id")),
                        "paper_id": md.get("paper_id") or md.get("row_identifier") or None,
                        "filename": md.get("filename") or md.get("study_title") or "",
                        "section": md.get("section") or "",
                        "domain": md.get("domain") or domain,
                        "content": md.get("chunk_text") or md.get("content") or md.get("chunk_text", ""),
                        "distance": float(m.get("score", 0)),
                    })

                return results

            else:
                # Simple query -> single combined_search call
                matches = await combined_search(query_text, index_name=index_name, top_n=top_k)
                results = []
                for m in matches:
                    md = m.get("metadata", {})
                    results.append({
                        "id": m.get("id"),
                        "chunk_id": md.get("chunk_id") or m.get("id"),
                        "paper_id": md.get("paper_id") or md.get("row_identifier") or None,
                        "filename": md.get("filename") or md.get("study_title") or "",
                        "section": md.get("section") or "",
                        "domain": md.get("domain") or domain,
                        "content": md.get("chunk_text") or md.get("content") or md.get("text") or "",
                        "distance": float(m.get("rerank_score", m.get("score", 0)))
                    })
                return results
        except Exception as e:
            logger.exception(f"Pinecone RAG path failed: {e}")

    # Fallback to existing Supabase Postgres search if Pinecone not configured or RAG path failed
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL (or DATABASE_URL) must be set")

    logger.debug(f"Generating embedding for query: '{query_text[:100]}...' (fallback Supabase path)")
    query_embedding = embed_text(query_text)
    if query_embedding is None:
        raise ValueError("Failed to generate embedding for query_text")

    emb_list = query_embedding.tolist()
    emb_literal = "[" + ", ".join(f"{x:.6f}" for x in emb_list) + "]"

    logger.debug(f"Embedding dimension: {len(emb_list)}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            func_check = await conn.fetchrow(
                "SELECT EXISTS (SELECT FROM information_schema.routines WHERE routine_name = 'search_research_chunks')"
            )
            
            if not func_check['exists']:
                return []
                
        except Exception as e:
            logger.error(f"Function check error: {e}", exc_info=True)
            return []
        try:
            try:
                rows = await conn.fetch(
                    """
                    SELECT 
                        id,
                        chunk_id,
                        paper_id,
                        filename,
                        section,
                        domain,
                        content,
                        distance,
                        pdf_url
                    FROM agent.search_research_chunks($1::vector, $2, $3)
                    """,
                    emb_literal,
                    top_k,
                    domain,
                )
            except Exception as inner_e:
                if "pdf_url" in str(inner_e) and "does not exist" in str(inner_e):
                    rows = await conn.fetch(
                        """
                        SELECT 
                            id,
                            chunk_id,
                            paper_id,
                            filename,
                            section,
                            domain,
                            content,
                            distance
                        FROM agent.search_research_chunks($1::vector, $2, $3)
                        """,
                        emb_literal,
                        top_k,
                        domain,
                    )
                else:
                    raise inner_e

            logger.info(f"SQL query returned {len(rows)} research chunks")
            
        except Exception as e:
            logger.error(f"SQL query error: {e}", exc_info=True)
            return []

    results: List[Dict[str, Any]] = []
    for r in rows:
        filename = r["filename"]
        if filename and filename.endswith('.json'):
            continue
            
        result_dict = {
            "id": r["id"],
            "chunk_id": r["chunk_id"],
            "paper_id": r["paper_id"],
            "filename": filename,
            "section": r["section"],
            "domain": r["domain"],
            "content": r["content"],
            "distance": float(r["distance"]),
        }
        
        if "pdf_url" in r.keys():
            result_dict["pdf_url"] = r["pdf_url"]
        
        results.append(result_dict)

    return results


def format_research_context(chunks: List[Dict[str, Any]], max_chars: int = 500) -> str:
    """Make chunks readable and prompt-friendly for Claude/Gemini."""
    lines: List[str] = []
    for c in chunks:
        lines.append(
            f"[chunk_id={c['chunk_id']}, paper_id={c.get('paper_id', '')}, "
            f"filename={c.get('filename', '')}, section={c.get('section', '')}, domain={c.get('domain', '')}]"
        )
        lines.append((c.get("content") or "")[:max_chars])
        lines.append("")
    return "\n".join(lines).strip()