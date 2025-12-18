"""
embeddings.py

Primary: Anthropic embeddings
Fallback: Google Generative AI embeddings
Final fallback: deterministic placeholder vector

Usage:
  from embeddings import embed_text
  vec = embed_text("some text")
  assert isinstance(vec, np.ndarray)
"""

import os
import logging
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

anthropic_client = None
try:
    from anthropic import Anthropic  
    _ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if _ANTHROPIC_API_KEY:
        anthropic_client = Anthropic(api_key=_ANTHROPIC_API_KEY)
    else:
        logger.warning("ANTHROPIC_API_KEY not found - Anthropic embeddings unavailable")
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    anthropic_client = None

# Optional Voyage client (preferred for dense embeddings)
voyage_client = None
try:
    import voyageai
    _VOYAGE_KEY = os.getenv("VOYAGE_API_KEY") or voyageai.api_key
    if _VOYAGE_KEY:
        try:
            voyage_client = voyageai.Client(api_key=_VOYAGE_KEY)
        except Exception:
            # older bindings may use voyageai.Embedding.create directly
            voyage_client = voyageai
    else:
        logger.debug("VOYAGE_API_KEY not set - Voyage embeddings unavailable")
except Exception as e:
    voyage_client = None
    logger.debug(f"Voyage client not available: {e}")

google_genai = None
try:
    import google.generativeai as genai  
    _G_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if _G_KEY:
        genai.configure(api_key=_G_KEY)
        google_genai = genai
    else:
        logger.warning("Google API key not found - Google embeddings unavailable")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")
    google_genai = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / (norm if norm > 0 else 1.0)

def _try_anthropic_embedding(text: str, model: str = "embed-1") -> Optional[np.ndarray]:
    """Try to generate embedding using Anthropic API"""
    if anthropic_client is None:
        return None
    try:
        if hasattr(anthropic_client, "embeddings") and hasattr(anthropic_client.embeddings, "create"):
            resp = anthropic_client.embeddings.create(model=model, input=text)
            if isinstance(resp, dict):
                emb = resp.get("data", [{}])[0].get("embedding") or resp.get("embedding")
            else:
                data = getattr(resp, "data", None)
                if isinstance(data, list) and data:
                    emb = getattr(data[0], "embedding", None) or (data[0].get("embedding") if isinstance(data[0], dict) else None)
                else:
                    emb = getattr(resp, "embedding", None)
            if emb:
                return _normalize(np.array(emb, dtype=np.float32))
        if hasattr(anthropic_client, "create_embedding"):
            resp = anthropic_client.create_embedding(text)
            if isinstance(resp, dict):
                emb = resp.get("data", [{}])[0].get("embedding") or resp.get("embedding")
            else:
                emb = getattr(resp, "embedding", None)
            if emb:
                return _normalize(np.array(emb, dtype=np.float32))
    except Exception as e:
        logger.debug(f"Anthropic embedding failed: {e}")
    return None


def _try_voyage_embedding(text: str, model: str = "voyage-3-large") -> Optional[np.ndarray]:
    """Try to generate embedding using Voyage client (preferred for dense semantics).

    Returns a normalized numpy array or None.
    """
    if voyage_client is None:
        return None
    try:
        # voyage_client may be a Client instance or the voyageai module
        if hasattr(voyage_client, "embed"):
            resp = voyage_client.embed(texts=[text], model=model, input_type="query")
            # resp may have .embeddings or .data
            emb = getattr(resp, "embeddings", None) or (resp.get("embeddings") if isinstance(resp, dict) else None)
            if emb and isinstance(emb, list) and emb:
                e = emb[0]
                return _normalize(np.array(e, dtype=np.float32))
        else:
            # fallback to module-level API
            resp = voyageai.Embedding.create(input=[text], model=model)
            # response shapes vary; try common fields
            if isinstance(resp, dict):
                data = resp.get("data") or resp.get("embeddings")
                if isinstance(data, list) and data:
                    emb = data[0].get("embedding") if isinstance(data[0], dict) else data[0]
                else:
                    emb = resp.get("embedding")
            else:
                emb = getattr(resp, "embedding", None)
            if emb is not None:
                return _normalize(np.array(emb, dtype=np.float32))
    except Exception as e:
        logger.debug(f"Voyage embedding failed: {e}")
    return None

def _try_google_embedding(text: str, model: str = "models/embedding-001", max_chars: int = 8000) -> Optional[np.ndarray]:
    """Try to generate embedding using Google Generative AI"""
    if google_genai is None:
        return None
    try:
        truncated = text[:max_chars]
        res = google_genai.embed_content(model=model, content=truncated, task_type="retrieval_document")
        if isinstance(res, dict):
            emb = res.get("embedding") or res.get("data", [{}])[0].get("embedding")
        else:
            emb = getattr(res, "embedding", None)
        if emb is not None:
            return _normalize(np.array(emb, dtype=np.float32))
    except Exception as e:
        logger.debug(f"Google embedding failed: {e}")
    return None


def embed_text(text: str, *,
               anthopic_model: str = "embed-1",
               google_model: str = "models/embedding-001",
               max_google_chars: int = 8000,
               placeholder_dim: int = 512) -> np.ndarray:
    """
    Return a normalized numpy.float32 embedding.
    Priority:
      1) Anthropic embeddings (if available)
      2) Google Generative AI embeddings (if available)
      3) Deterministic placeholder vector of dimension `placeholder_dim`
    """
    if not isinstance(text, str):
        text = str(text or "")

    # Priority: Voyage (dense) -> Anthropic -> Google -> placeholder
    emb = _try_voyage_embedding(text, model="voyage-3-large")
    if emb is not None:
        return emb

    emb = _try_anthropic_embedding(text, model=anthopic_model)
    if emb is not None:
        return emb

    emb = _try_google_embedding(text, model=google_model, max_chars=max_google_chars)
    if emb is not None:
        return emb

    logger.debug("Using placeholder embedding (all providers unavailable)")
    rng = np.random.RandomState(0)
    vec = rng.randn(placeholder_dim).astype(np.float32)
    return _normalize(vec)


async def embed_text_batch(texts: list[str], 
                           batch_size: int = 10,
                           **kwargs) -> list[np.ndarray]:
    """
    Embed multiple texts with concurrent processing.
    Processes texts in batches to avoid overwhelming the API.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of concurrent embeddings to process at once
        **kwargs: Additional arguments passed to embed_text
    
    Returns:
        List of numpy arrays, one embedding per input text
    """
    if not texts:
        return []
    
    logger.debug(f"Batch embedding {len(texts)} texts with batch_size={batch_size}")
    
    def _embed_sync(text: str) -> np.ndarray:
        return embed_text(text, **kwargs)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [
                loop.run_in_executor(executor, _embed_sync, text)
                for text in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
    
    logger.debug(f"Completed batch embedding of {len(results)} texts")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = "Test embedding for RAG pipeline."
    v = embed_text(s)
    logger.info(f"Embedding shape: {v.shape}")
