#!/usr/bin/env python3
"""
Run a single-analysis RAG flow from CLI:
 - combined_search (dense + sparse)
 - build numbered research context with REF ID [1], APA citations
 - call safe_generate (Anthropic/Gemini) to produce final JSON analysis

Usage:
  python scripts/single_analysis_rag.py --query "what works to increase school attendance"

"""
import os
import argparse
import asyncio
import json
import logging

from app.services.pinecone_rag import combined_search

logger = logging.getLogger(__name__)


def format_apa(metadata: dict) -> str:
    # Prefer explicit full_citation
    if not metadata:
        return ""
    fc = metadata.get("full_citation") or metadata.get("Full Citation")
    if fc:
        return fc

    authors = metadata.get("authors") or metadata.get("author") or ""
    year = metadata.get("year") or metadata.get("Year") or "n.d."
    title = metadata.get("study_title") or metadata.get("Study Title") or metadata.get("filename") or ""
    src = metadata.get("source") or metadata.get("journal") or ""
    parts = []
    if authors:
        parts.append(str(authors))
    parts.append(f"({year})")
    if title:
        parts.append(str(title))
    if src:
        parts.append(str(src))
    return " ".join([p for p in parts if p])


def build_numbered_context(matches: list) -> str:
    parts = []
    for idx, m in enumerate(matches, start=1):
        md = m.get("metadata") or {}
        title = md.get("study_title") or md.get("filename") or "Untitled"
        citation = format_apa(md)
        link = md.get("study_link") or md.get("gdrive_link") or md.get("link") or ""
        content = (md.get("chunk_text") or md.get("content") or md.get("citation_text") or "")[:2000]
        block = (
            f"### REF ID [{idx}]\n"
            f"TITLE: {title}\n"
            f"FULL CITATION: {citation}\n"
            f"LINK: {link}\n"
            f"CONTENT EXCERPTS:\n{content}\n"
        )
        parts.append(block)
    return "\n\n".join(parts)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--index", type=str, default=os.getenv("PINECONE_INDEX_NAME"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    if not args.index:
        print("Pinecone index not configured. Set --index or PINECONE_INDEX_NAME env var.")
        return

    matches = await combined_search(args.query, index_name=args.index, top_n=args.top_n)

    if not matches:
        print("No matches returned. Check Pinecone configuration.")
        return

    # Build numbered context
    context = build_numbered_context(matches)

    # Build final system + user prompt (simple wrapper) - reuse analysis.safe_generate if available
    try:
        from app.services.analysis import safe_generate
    except Exception:
        safe_generate = None

    system_prompt = "You are a research analyst. Use the provided research context to answer the user query and include inline citations like [1]. Return ONLY valid JSON with keys: program_summary, content_analysis, strengths_and_opportunities, recommendations, citations"
    user_message = f"USER QUERY: {args.query}\n\nRESEARCH CONTEXT:\n{context}"

    if safe_generate:
        raw = safe_generate(system_prompt, user_message, max_tokens=2000)
        print(raw)
    else:
        print("No safe_generate available. Here is the assembled prompt you can use with your LLM:\n")
        print(system_prompt)
        print("\n---USER MESSAGE---\n")
        print(user_message[:8000])


if __name__ == "__main__":
    asyncio.run(main())
