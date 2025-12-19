"""
Enhanced Chat Service for Single Analysis Follow-ups

Supports:
- Text messages (questions/instructions)
- File uploads (PDF, DOCX, TXT) 
- Image uploads (OCR extraction)
- Hybrid RAG integration for relevant citations

Flow:
    User Input (text + files + images)
           ↓
    Extract content from all inputs
           ↓
    Combine with original analysis context
           ↓
    Hybrid RAG search (if needed)
           ↓
    Generate response with citations
"""

import os
import io
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Import dependencies
from app.services.comparative_analysis import safe_generate, extract_text_from_upload
from app.services.single_analysis_rag import (
    generate_with_rag_citations,
    generate_from_precomputed_candidates,
    check_query_relevance,
    MIN_RELEVANCE_SCORE
)

# Try to import hybrid retrieval
try:
    from app.services.hybrid_retrieval import (
        hybrid_retrieval_only,
        is_document as check_is_document
    )
    HYBRID_RETRIEVAL_AVAILABLE = True
    logger.info("Hybrid retrieval available for chat")
except ImportError:
    HYBRID_RETRIEVAL_AVAILABLE = False
    logger.warning("Hybrid retrieval not available for chat")

# Try to import OCR for images
try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("OCR (pytesseract) available for image extraction")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - image OCR disabled")

# Try to import pdfplumber for PDF extraction
try:
    import pdfplumber
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    logger.warning("pdfplumber not available - PDF extraction limited")


async def extract_text_from_image(image_file: UploadFile) -> str:
    """
    Extract text from an uploaded image using OCR.
    
    Supports: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        Extracted text from image, or empty string if extraction fails
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("OCR not available - cannot extract text from image")
        return ""
    
    try:
        content = await image_file.read()
        await image_file.seek(0)  # Reset for potential re-read
        
        # Open image with PIL
        image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if necessary (for RGBA images)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Convert to grayscale for better OCR
        image_gray = image.convert('L')
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image_gray)
        
        if extracted_text and extracted_text.strip():
            logger.info(f"Extracted {len(extracted_text)} chars from image: {image_file.filename}")
            return extracted_text.strip()
        else:
            logger.warning(f"No text found in image: {image_file.filename}")
            return ""
            
    except Exception as e:
        logger.error(f"Image OCR extraction failed for {image_file.filename}: {e}")
        return ""


async def extract_content_from_upload(upload: UploadFile) -> Dict[str, Any]:
    """
    Extract content from any uploaded file (document or image).
    
    Supports:
    - Documents: PDF, DOCX, DOC, TXT
    - Images: PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF
    
    Args:
        upload: Uploaded file
        
    Returns:
        Dict with:
            - content: Extracted text
            - type: 'document' or 'image'
            - filename: Original filename
            - success: Whether extraction succeeded
    """
    filename = upload.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    # Image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
    
    # Document extensions
    doc_extensions = {'.pdf', '.docx', '.doc', '.txt', '.text', '.md', '.rtf'}
    
    if ext in image_extensions:
        # Extract from image using OCR
        content = await extract_text_from_image(upload)
        return {
            "content": content,
            "type": "image",
            "filename": filename,
            "success": bool(content),
            "length": len(content)
        }
    elif ext in doc_extensions:
        # Extract from document
        content = await extract_text_from_upload(upload)
        return {
            "content": content,
            "type": "document", 
            "filename": filename,
            "success": bool(content),
            "length": len(content)
        }
    else:
        # Try to read as text
        try:
            content_bytes = await upload.read()
            await upload.seek(0)
            content = content_bytes.decode('utf-8', errors='ignore')
            return {
                "content": content,
                "type": "text",
                "filename": filename,
                "success": bool(content),
                "length": len(content)
            }
        except Exception as e:
            logger.error(f"Failed to extract content from {filename}: {e}")
            return {
                "content": "",
                "type": "unknown",
                "filename": filename,
                "success": False,
                "length": 0
            }


async def generate_single_analysis_chat_response(
    user_question: str,
    analysis_data: Dict[str, Any],
    domain: Optional[str] = None,
    top_k: int = 5,
    max_tokens: int = 2000,
    files: Optional[List[UploadFile]] = None,
    use_hybrid_rag: bool = True
) -> Dict[str, Any]:
    """
    Generate a chat response based on single analysis data with RAG support.
    
    NOW SUPPORTS:
    - Text questions/instructions
    - File uploads (PDF, DOCX, TXT)
    - Image uploads (OCR extraction)
    - Hybrid RAG for relevant citations
    
    Args:
        user_question: The user's follow-up question or instruction
        analysis_data: The original analysis result containing:
            - response: Original analysis text
            - chat_type: Type of analysis performed
            - citations: Research citations used
            - file_metadata: Files that were analyzed
            - content_analysis: Content statistics
        domain: Optional domain filter for RAG search
        top_k: Number of research chunks to retrieve
        max_tokens: Maximum tokens in response
        files: Optional list of uploaded files (documents or images)
        use_hybrid_rag: Whether to use hybrid RAG (default True)
        
    Returns:
        Dict containing answer, citations, and context used
    """
    logger.info("=" * 60)
    logger.info("SINGLE ANALYSIS CHAT - ENHANCED")
    logger.info("=" * 60)
    
    original_response = analysis_data.get("response", "")
    chat_type = analysis_data.get("chat_type", "ANALYSIS")
    file_metadata = analysis_data.get("file_metadata", [])
    content_analysis = analysis_data.get("content_analysis", {})
    original_citations = analysis_data.get("citations", [])
    analysis_id = analysis_data.get("analysis_id", "unknown")
    
    logger.info(f"Original Analysis ID: {analysis_id}")
    logger.info(f"Chat Type: {chat_type}")
    logger.info(f"User Question: {user_question[:100]}...")
    logger.info(f"Files Uploaded: {len(files) if files else 0}")
    
    # =========================================================
    # STEP 1: Extract content from uploaded files/images
    # =========================================================
    uploaded_content = []
    if files:
        logger.info(f"[1/4] Extracting content from {len(files)} uploaded files...")
        for f in files:
            extraction = await extract_content_from_upload(f)
            if extraction["success"]:
                uploaded_content.append(extraction)
                logger.info(f"  ✓ {extraction['filename']} ({extraction['type']}): {extraction['length']} chars")
            else:
                logger.warning(f"  ✗ Failed to extract from: {extraction['filename']}")
    
    # Build context from uploaded content
    uploaded_context = ""
    if uploaded_content:
        uploaded_context = "\n\n## NEW UPLOADED CONTENT\n\n"
        for i, content in enumerate(uploaded_content, 1):
            uploaded_context += f"### Upload {i}: {content['filename']} ({content['type']})\n"
            uploaded_context += content['content'][:3000]  # Limit per file
            if len(content['content']) > 3000:
                uploaded_context += "\n[... content truncated ...]"
            uploaded_context += "\n\n"
    
    # =========================================================
    # STEP 2: Build combined query for RAG search
    # =========================================================
    logger.info("[2/4] Building combined query...")
    
    # Combine user question with uploaded content for RAG search
    if uploaded_content:
        combined_query = f"""
User Question: {user_question}

Context from uploaded content:
{' '.join([c['content'][:500] for c in uploaded_content])}
"""
    else:
        combined_query = user_question
    
    # =========================================================
    # STEP 3: Hybrid RAG Search (if enabled)
    # =========================================================
    rag_context = ""
    rag_citations = []
    candidates = []
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if use_hybrid_rag and index_name and HYBRID_RETRIEVAL_AVAILABLE:
        logger.info("[3/4] Performing hybrid RAG search...")
        try:
            retrieval_result = await hybrid_retrieval_only(
                query_text=combined_query,
                index_name=index_name,
                top_k_retrieval=30,
                top_n_rerank=top_k,
                chunk_documents=False,  # Don't chunk for chat queries
                namespace="research"
            )
            
            candidates = retrieval_result.get("top_results", [])
            rag_context = retrieval_result.get("context", "")
            rag_citations = retrieval_result.get("citations", [])
            
            logger.info(f"  → Retrieved {len(candidates)} candidates")
            
            # Check relevance
            if candidates:
                relevance_check = check_query_relevance(combined_query, candidates)
                logger.info(f"  → Relevance: avg={relevance_check['avg_score']:.2f}, relevant={relevance_check['is_relevant']}")
                
                if not relevance_check["is_relevant"]:
                    logger.warning("  → Low relevance detected, citations may not be directly applicable")
                    
        except Exception as e:
            logger.error(f"Hybrid RAG search failed: {e}")
            import traceback
            traceback.print_exc()
    elif use_hybrid_rag and index_name:
        # Fallback to non-hybrid search
        logger.info("[3/4] Using fallback RAG search...")
        try:
            from app.services.pinecone_rag import combined_search
            matches = await combined_search(
                query_text=combined_query,
                index_name=index_name,
                top_k=top_k * 2,
                top_n=top_k,
                rerank=True
            )
            candidates = matches[:top_k] if matches else []
            logger.info(f"  → Retrieved {len(candidates)} candidates via fallback")
        except Exception as e:
            logger.warning(f"Fallback RAG search failed: {e}")
    else:
        logger.info("[3/4] RAG search skipped (not configured)")
    
    # =========================================================
    # STEP 4: Generate Response
    # =========================================================
    logger.info("[4/4] Generating response...")
    
    # Build enhanced system prompt
    system_prompt = f"""You are an expert analysis assistant helping with follow-up questions about a previously completed {chat_type} analysis.

## Original Analysis Context

The user previously uploaded {len(file_metadata)} file(s) for analysis:
{_format_file_list(file_metadata)}

The analysis was {content_analysis.get('total_words', 'unknown')} words and contained detailed insights.

## Your Role

1. Answer the user's follow-up question based on:
   - The original analysis findings (provided below)
   - Any NEW content the user has uploaded in this message
   - Additional research evidence from the knowledge base
   - Your expertise in philanthropy and nonprofit evaluation

2. **CITATION FORMAT**: Use REF ID citations like [1], [2], [3] when referencing research.
   - Example: "Studies show significant impact [1, 2]."
   - Only cite sources that are actually provided in the research context.

3. Reference specific points from the original analysis when relevant.

4. If the user uploaded new documents or images, analyze their content and relate it to the original analysis.

5. Be conversational but thorough - this is a follow-up, not a full report.

## Original Analysis Summary
{_extract_key_findings(original_response)}
{uploaded_context}
---

Now answer the user's follow-up question with evidence-based insights."""

    # If we have candidates, use the precomputed generation
    if candidates and len(candidates) > 0:
        try:
            # Format candidates for generation
            formatted_candidates = []
            for c in candidates:
                md = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else c
                formatted_candidates.append({
                    "id": c.get("id"),
                    "chunk_id": md.get("chunk_id") or c.get("id"),
                    "content": md.get("chunk_text") or md.get("content") or c.get("content", ""),
                    "study_title": md.get("study_title") or "",
                    "full_citation": md.get("full_citation") or "",
                    "year": md.get("year") or "n.d.",
                    "metadata": md
                })
            
            result = await generate_from_precomputed_candidates(
                system_prompt=system_prompt,
                user_query=user_question,
                candidates=formatted_candidates,
                max_tokens=max_tokens,
                enforce_relevance=True  # Check relevance
            )
        except Exception as e:
            logger.warning(f"Generation from candidates failed: {e}, using fallback")
            result = await generate_with_rag_citations(
                system_prompt=system_prompt,
                user_query=user_question,
                top_k_research=top_k,
                domain=domain,
                max_tokens=max_tokens,
                enable_web_fallback=False
            )
    else:
        # Use standard RAG generation
        result = await generate_with_rag_citations(
            system_prompt=system_prompt,
            user_query=user_question,
            top_k_research=top_k,
            domain=domain,
            max_tokens=max_tokens,
            enable_web_fallback=False
        )
    
    # Add chat-specific metadata
    result["original_analysis_type"] = chat_type
    result["original_analysis_id"] = analysis_id
    result["files_analyzed"] = [f["filename"] for f in file_metadata]
    result["is_follow_up"] = True
    result["uploaded_in_chat"] = [c["filename"] for c in uploaded_content] if uploaded_content else []
    result["uploaded_content_types"] = [c["type"] for c in uploaded_content] if uploaded_content else []
    
    logger.info(f"Chat response generated: {len(result.get('response', ''))} chars")
    
    return result


def _format_file_list(file_metadata: list) -> str:
    """Format file list for context"""
    if not file_metadata:
        return "No files"
    
    lines = []
    for i, file in enumerate(file_metadata, 1):
        filename = file.get("filename", "Unknown")
        length = file.get("length", 0)
        lines.append(f"{i}. {filename} ({length:,} characters)")
    
    return "\n".join(lines)


def _extract_key_findings(original_response: str, max_chars: int = 2000) -> str:
    """Extract key findings from original analysis for context"""
    
    if len(original_response) <= max_chars:
        return original_response
    
    lines = original_response.split('\n')
    
    key_sections = []
    current_section = []
    total_chars = 0
    
    for line in lines:
        line_chars = len(line) + 1 
        
        if total_chars + line_chars > max_chars:
            break
        
        current_section.append(line)
        total_chars += line_chars
        
        if line.startswith('#') and current_section:
            key_sections.extend(current_section)
            current_section = []
    
    if current_section:
        key_sections.extend(current_section)
    
    result = '\n'.join(key_sections)
    
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[... additional analysis omitted for brevity ...]"
    
    return result


async def generate_simple_chat_response(
    user_question: str,
    analysis_data: Dict[str, Any],
    files: Optional[List[UploadFile]] = None
) -> Dict[str, Any]:
    """
    Generate a simple chat response without RAG (fallback).
    
    Uses only the original analysis data and uploaded content to answer questions.
    """
    
    original_response = analysis_data.get("response", "")
    chat_type = analysis_data.get("chat_type", "ANALYSIS")
    file_metadata = analysis_data.get("file_metadata", [])
    
    # Extract content from uploads if provided
    uploaded_context = ""
    uploaded_files_info = []
    if files:
        for f in files:
            extraction = await extract_content_from_upload(f)
            if extraction["success"]:
                uploaded_context += f"\n\n### {extraction['filename']}:\n{extraction['content'][:2000]}"
                uploaded_files_info.append(extraction['filename'])
    
    system_msg = f"""You are an expert analysis assistant. Answer the user's question based on the analysis and any new content provided below.

## Context:
- Analysis Type: {chat_type}
- Files Analyzed: {', '.join([f.get('filename', 'Unknown') for f in file_metadata])}

## Original Analysis:
{original_response[:4000]}

{f'## New Uploaded Content:{uploaded_context}' if uploaded_context else ''}

---

Answer the user's question clearly and concisely, referencing specific findings from the analysis."""

    user_msg = f"Question: {user_question}"
    
    answer = safe_generate(system_msg, user_msg, max_tokens=1500)
    
    if not answer:
        return {
            "response": "I apologize, but I'm having trouble generating a response. Please try again.",
            "citations": [],
            "num_sources": 0,
            "has_research": False,
            "is_follow_up": True
        }
    
    return {
        "response": answer,
        "citations": [],
        "num_sources": 0,
        "has_research": False,
        "is_follow_up": True,
        "original_analysis_type": chat_type,
        "uploaded_in_chat": uploaded_files_info
    }
