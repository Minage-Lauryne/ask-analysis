"""
Single Analysis Route with File Uploads and RAG Integration
Provides analysis with automatic research citations for uploaded files
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
from app.models.schemas import SingleAnalysisResponse, Citation
from app.services.single_analysis_service import single_analysis_service
from app.services.single_analysis_chat import generate_single_analysis_chat_response
from app.database import get_single_analysis
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/files", response_model=SingleAnalysisResponse)
async def analyze_files_with_rag(
    request: Request,
    user_query: Optional[str] = Form(None, description="Optional text/question for analysis"),
    chat_type: str = Form("ANALYSIS", description="Type of analysis"),
    domain: Optional[str] = Form(None, description="Optional domain filter"),  
    top_k: int = Form(10, description="Number of research sources"),
    max_tokens: int = Form(4000, description="Maximum response length"),
    organization_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    initial_analysis_context: Optional[str] = Form(None, description="Context from initial analysis"),
):
    """
    Unified single analysis endpoint supporting multiple input modes:
    
    1. **Files only**: Upload documents for analysis
    2. **Text only**: Submit a question or text for analysis (via user_query)
    3. **Files + Text**: Upload documents AND ask a specific question about them
    4. **Context-aware**: Any of the above + initial_analysis_context for specialized follow-up
    
    At least one of 'files' or 'user_query' must be provided.
    """
    
    try:
        logger.info("Single analysis request received")
        form = await request.form()
        files_raw = form.getlist("files")
        
        logger.debug(f"Files extraction - count: {len(files_raw) if files_raw else 0}")
        if files_raw:
            for idx, f in enumerate(files_raw):
                logger.debug(f"File {idx}: type={type(f)}, value={repr(f)[:100]}")
        
        valid_files = []
        if files_raw:
            for f in files_raw:
                if hasattr(f, 'filename') and hasattr(f, 'read'):
                    try:
                        content = await f.read(10)
                        await f.seek(0)
                        if f.filename and f.filename.strip() != "" and len(content) > 0:
                            valid_files.append(f)
                        else:
                            logger.warning(f"Skipping empty/invalid file: {f.filename}")
                    except Exception as e:
                        logger.error(f"Error reading file {getattr(f, 'filename', 'unknown')}: {e}")
                        continue
                elif isinstance(f, str):
                    logger.debug(f"Skipping string placeholder in files: {f}")
                    continue
                else:
                    logger.debug(f"Unknown file type: {type(f)} - {repr(f)[:100]}")
        
        logger.info(f"Valid files: {len(valid_files)}, query provided: {bool(user_query)}")
        
        if not valid_files and not user_query:
            logger.error("Request missing both files and user_query")
            raise HTTPException(
                status_code=400, 
                detail="At least one of 'files' or 'user_query' must be provided"
            )
        
        actual_domain = None
        if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
            actual_domain = domain.strip()
        
        actual_org_id = None
        if organization_id and organization_id.strip() and organization_id.lower() not in ["none", "null", "string"]:
            actual_org_id = organization_id.strip()
        
        actual_user_id = None
        if user_id and user_id.strip() and user_id.lower() not in ["none", "null", "string"]:
            actual_user_id = user_id.strip()
        
        actual_initial_context = None
        if initial_analysis_context and initial_analysis_context.strip() and initial_analysis_context.lower() not in ["none", "null", "string"]:
            actual_initial_context = initial_analysis_context.strip()
        
        actual_user_query = None
        if user_query and user_query.strip() and user_query.lower() not in ["none", "null", "string"]:
            actual_user_query = user_query.strip()
        
        logger.info(f"Analysis request - Files: {len(valid_files)}, Query: {'Yes' if actual_user_query else 'No'}, Type: {chat_type}")
        if actual_user_query:
            logger.debug(f"Query preview: {actual_user_query[:200]}...")
        if actual_initial_context:
            logger.info(f"Initial context provided: {len(actual_initial_context)} chars")
            logger.debug(f"Context preview: {actual_initial_context[:200]}...")
        
        from app.services.prompts import CHAT_TYPES
        valid_chat_types = list(CHAT_TYPES.keys())
        
        if chat_type not in valid_chat_types:
            logger.error(f"Invalid chat type: {chat_type}. Valid types: {', '.join(valid_chat_types)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid chat type. Must be one of: {', '.join(valid_chat_types)}"
            )
        
        if valid_files:
            logger.info(f"Processing files-based analysis with {len(valid_files)} files")
            result = await single_analysis_service.analyze_files(
                files=valid_files,
                user_query=actual_user_query,
                chat_type=chat_type,
                domain=actual_domain,  
                top_k=top_k,
                max_tokens=max_tokens,
                organization_id=actual_org_id,
                user_id=actual_user_id,
                enable_web_fallback=False,
                initial_analysis_context=actual_initial_context
            )
        else:
            logger.info("Processing text-only analysis")
            from app.services.single_analysis_rag import generate_with_rag_citations
            
            result = await generate_with_rag_citations(
                system_prompt=_get_system_prompt_for_chat_type(chat_type),
                user_query=actual_user_query,
                top_k_research=top_k,
                domain=actual_domain,
                max_tokens=max_tokens,
                enable_web_fallback=False
            )
            
            result['analysis_id'] = str(uuid.uuid4())
            result['chat_type'] = chat_type
            result['file_metadata'] = []
            result['content_analysis'] = {}
            result['analysis_mode'] = 'text_only'
            result['user_query'] = actual_user_query
            
            try:
                from app.database import insert_single_analysis
                logger.info(f"Storing text-only analysis: {result['analysis_id']}")
                insert_single_analysis(
                    analysis_id=result['analysis_id'],
                    chat_type=chat_type,
                    response_text=result['response'],
                    citations=result['citations'],
                    file_metadata=[],
                    content_analysis={},
                    organization_id=actual_org_id,
                    user_id=actual_user_id
                )
            except Exception as e:
                logger.error(f"Failed to store text-only analysis: {e}")
        
        citations = [
            Citation(
                id=cit.get('id', idx),
                chunk_id=cit.get('chunk_id', ''),
                paper_id=cit.get('paper_id', ''),
                filename=cit.get('filename', ''),
                section=cit.get('section', ''),
                domain=cit.get('domain', ''),
                content=cit.get('content', ''),
                distance=cit.get('distance', 0.0)
            )
            for idx, cit in enumerate(result['citations'])
        ]
        
        return SingleAnalysisResponse(
            answer=result['response'],
            citations=citations,
            has_research=result['has_research'],
            num_sources=result['num_sources'],
            chat_type=chat_type,
            file_metadata=result.get('file_metadata', []),
            content_analysis=result.get('content_analysis', {}),
            analysis_id=result.get('analysis_id'),
            organization_id=actual_org_id,
            user_id=actual_user_id,
            context_used=result.get('context_used', False),
            context_aware=result.get('context_aware', False),
            analysis_mode=result.get('analysis_mode'),
            user_query=result.get('user_query')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/", response_model=SingleAnalysisResponse)
async def analyze_single_with_rag_text(
    message: str = Form(..., description="Analysis request text"),
    chat_type: str = Form("ANALYSIS", description="Type of analysis"),
    domain: Optional[str] = Form(None, description="Optional domain filter"),
    top_k: int = Form(10, description="Number of research sources"),
    max_tokens: int = Form(4000, description="Maximum response length"),
    organization_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    """
    DEPRECATED: Text-based single analysis endpoint 
    
    Maintained for backward compatibility. 
    **New integrations should use POST /single-analysis/files with 'message' parameter instead.**
    
    That endpoint supports files, text, or both.
    """
    from app.services.single_analysis_rag import generate_with_rag_citations
    
    try:
        actual_domain = None
        if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
            actual_domain = domain.strip()
        
        actual_org_id = None
        if organization_id and organization_id.strip() and organization_id.lower() not in ["none", "null", "string"]:
            actual_org_id = organization_id.strip()
        
        actual_user_id = None
        if user_id and user_id.strip() and user_id.lower() not in ["none", "null", "string"]:
            actual_user_id = user_id.strip()
        
        result = await generate_with_rag_citations(
            system_prompt=_get_system_prompt_for_chat_type(chat_type),
            user_query=message,
            top_k_research=top_k,
            domain=actual_domain,  
            max_tokens=max_tokens
        )
        
        citations = [
            Citation(
                id=cit['id'],
                chunk_id=cit['chunk_id'],
                paper_id=cit['paper_id'],
                filename=cit['filename'],
                section=cit['section'],
                domain=cit['domain'],
                content=cit['content'],
                distance=cit['distance']
            )
            for cit in result['citations']
        ]
        
        return SingleAnalysisResponse(
            answer=result['response'],
            citations=citations,
            has_research=result['has_research'],
            num_sources=result['num_sources'],
            chat_type=chat_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _get_system_prompt_for_chat_type(chat_type: str) -> str:
    """Get system prompt for text-based analysis using centralized prompts"""
    from app.services.prompts import get_prompt_by_chat_type
    
    return get_prompt_by_chat_type(
        chat_type=chat_type,
        is_first_message=True,
        has_initial_context=False
    )


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "single-analysis-rag",
        "rag_enabled": True,
        "file_uploads_supported": True
    }


@router.get("/chat-types")
async def get_available_chat_types():
    from app.services.prompts import CHAT_TYPES, PROMPT_MAP
    
    chat_types_list = []
    for value, label in CHAT_TYPES.items():
        prompt_info = PROMPT_MAP.get(value, {})
        chat_types_list.append({
            "value": value,
            "label": label,
            "description": prompt_info.get("description", ""),
            "requires_rag": prompt_info.get("requires_rag", True),
            "with_context": prompt_info.get("with_context", True)
        })
    
    return {"chat_types": chat_types_list}


@router.post("/chat")
async def chat_with_analysis(
    analysis_id: str = Form(..., description="ID of the analysis to chat about"),
    message: str = Form(..., description="User's follow-up question"),
    domain: Optional[str] = Form(None, description="Optional domain filter for research"),
    top_k: int = Form(5, description="Number of research sources to retrieve"),
    max_tokens: int = Form(2000, description="Maximum response length"),
    organization_id: Optional[str] = Form(None, description="Organization ID for access control"),
    user_id: Optional[str] = Form(None, description="User ID for access control")
):
    """
    BASIC Chat endpoint for follow-up questions about a single analysis (TEXT ONLY)
    
    For file/image uploads, use the ENHANCED endpoint: POST /single-analysis/chat/v2
    
    This allows users to:
    1. Ask follow-up questions about their analysis
    2. Get additional research evidence
    3. Explore specific aspects in more detail
    
    The analysis_id should match the analysis_id returned from /single-analysis/files
    
    Example:
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat" \
      -F "analysis_id=abc-123-def" \
      -F "message=Can you explain more about the evidence for this approach?" \
      -F "user_id=user-123"
    ```
    
    Requires the analysis to have been stored in the database (pass user_id or organization_id to /files endpoint)
    """
    
    actual_user_id = None
    if user_id and user_id.strip() and user_id.lower() not in ["none", "null", "string"]:
        actual_user_id = user_id.strip()
    
    actual_org_id = None
    if organization_id and organization_id.strip() and organization_id.lower() not in ["none", "null", "string"]:
        actual_org_id = organization_id.strip()
    
    logger.debug(f"Processed IDs - user_id: {repr(actual_user_id)}, org_id: {repr(actual_org_id)}")
    
    try:
        logger.info(f"Looking up analysis: {analysis_id}")
        analysis_data = get_single_analysis(
            analysis_id=analysis_id,
            organization_id=actual_org_id,
            user_id=actual_user_id
        )
        
        if not analysis_data:
            logger.error(f"Analysis not found in database: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Analysis '{analysis_id}' not found. "
                    "Make sure the analysis_id is correct and that you have access to it. "
                    "The analysis must have been created with a user_id or organization_id to be retrievable."
                )
            )
        
        logger.info(f"Found analysis: {analysis_data.get('chat_type', 'UNKNOWN')}, response length: {len(analysis_data.get('response', ''))} chars")
        
        actual_domain = None
        if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
            actual_domain = domain.strip()
        
        result = await generate_single_analysis_chat_response(
            user_question=message,
            analysis_data=analysis_data,
            domain=actual_domain,
            top_k=top_k,
            max_tokens=max_tokens
        )
        
        logger.info(f"Chat response generated: {len(result['response'])} chars, citations: {result['num_sources']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chat response: {str(e)}"
        )


@router.post("/chat/v2")
async def enhanced_chat_with_analysis(
    request: Request,
    analysis_id: str = Form(..., description="ID of the analysis to chat about"),
    message: str = Form("", description="User's follow-up question or instruction"),
    domain: Optional[str] = Form(None, description="Optional domain filter for research"),
    top_k: int = Form(5, description="Number of research sources to retrieve"),
    max_tokens: int = Form(2000, description="Maximum response length"),
    organization_id: Optional[str] = Form(None, description="Organization ID for access control"),
    user_id: Optional[str] = Form(None, description="User ID for access control"),
    use_hybrid_rag: bool = Form(True, description="Use hybrid RAG for citations")
):
    """
    ENHANCED Chat endpoint supporting FILES + IMAGES + TEXT
    
    This is the recommended endpoint for follow-up interactions as it supports:
    
    1. **Text messages** - Questions or instructions
    2. **Document uploads** - PDF, DOCX, DOC, TXT files (content extracted)
    3. **Image uploads** - PNG, JPG, JPEG, GIF, etc. (OCR text extraction)
    4. **Combined inputs** - Any combination of the above
    
    The hybrid RAG system searches for relevant research citations based on:
    - Your follow-up question
    - Content from any uploaded files/images
    - Context from the original analysis
    
    **File Input Options:**
    - `files[]` - Array of files (documents or images)
    
    Example with text only:
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat/v2" \
      -F "analysis_id=abc-123-def" \
      -F "message=What does this mean for implementation?" \
      -F "user_id=user-123"
    ```
    
    Example with file upload:
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat/v2" \
      -F "analysis_id=abc-123-def" \
      -F "message=How does this compare to our original analysis?" \
      -F "files[]=@new_document.pdf" \
      -F "user_id=user-123"
    ```
    
    Example with image upload (OCR):
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat/v2" \
      -F "analysis_id=abc-123-def" \
      -F "message=What does this screenshot show about outcomes?" \
      -F "files[]=@table_screenshot.png" \
      -F "user_id=user-123"
    ```
    
    Response includes:
    - response: The generated answer with citations
    - citations: List of research sources used
    - uploaded_in_chat: List of files processed in this message
    - is_follow_up: Always True for chat responses
    """
    
    logger.info("=" * 60)
    logger.info("ENHANCED CHAT ENDPOINT V2")
    logger.info("=" * 60)
    
    # Extract files from form
    form = await request.form()
    files_raw = form.getlist("files") or form.getlist("files[]")
    
    valid_files = []
    if files_raw:
        for f in files_raw:
            if hasattr(f, 'filename') and hasattr(f, 'read'):
                try:
                    # Check if file has content
                    content = await f.read(10)
                    await f.seek(0)
                    if f.filename and f.filename.strip() != "" and len(content) > 0:
                        valid_files.append(f)
                        logger.info(f"  File accepted: {f.filename}")
                    else:
                        logger.warning(f"  Skipping empty file: {f.filename}")
                except Exception as e:
                    logger.error(f"  Error reading file: {e}")
            elif isinstance(f, str):
                logger.debug(f"  Skipping string placeholder: {f}")
    
    logger.info(f"Chat request - Analysis: {analysis_id}, Files: {len(valid_files)}, Message: {len(message)} chars")
    
    # Validate inputs
    if not message.strip() and not valid_files:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'message' or 'files' must be provided for chat"
        )
    
    # Process optional parameters
    actual_user_id = None
    if user_id and user_id.strip() and user_id.lower() not in ["none", "null", "string"]:
        actual_user_id = user_id.strip()
    
    actual_org_id = None
    if organization_id and organization_id.strip() and organization_id.lower() not in ["none", "null", "string"]:
        actual_org_id = organization_id.strip()
    
    actual_domain = None
    if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
        actual_domain = domain.strip()
    
    actual_message = message.strip() if message else ""
    
    try:
        # Retrieve the original analysis
        logger.info(f"Looking up analysis: {analysis_id}")
        analysis_data = get_single_analysis(
            analysis_id=analysis_id,
            organization_id=actual_org_id,
            user_id=actual_user_id
        )
        
        if not analysis_data:
            logger.error(f"Analysis not found: {analysis_id}")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Analysis '{analysis_id}' not found. "
                    "Make sure the analysis_id is correct and that you have access to it. "
                    "The analysis must have been created with a user_id or organization_id to be retrievable."
                )
            )
        
        logger.info(f"Found analysis: {analysis_data.get('chat_type', 'UNKNOWN')}")
        
        # Generate chat response with the enhanced service
        result = await generate_single_analysis_chat_response(
            user_question=actual_message if actual_message else "Please analyze the uploaded content.",
            analysis_data=analysis_data,
            domain=actual_domain,
            top_k=top_k,
            max_tokens=max_tokens,
            files=valid_files if valid_files else None,
            use_hybrid_rag=use_hybrid_rag
        )
        
        logger.info(f"Chat response generated: {len(result.get('response', ''))} chars")
        logger.info(f"  → Citations: {result.get('num_sources', 0)}")
        logger.info(f"  → Uploaded processed: {result.get('uploaded_in_chat', [])}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced chat generation failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chat response: {str(e)}"
        )


@router.post("/chat/standalone")
async def standalone_chat(
    request: Request,
    message: str = Form("", description="User's question or instruction"),
    domain: Optional[str] = Form(None, description="Optional domain filter for research"),
    top_k: int = Form(10, description="Number of research sources to retrieve"),
    max_tokens: int = Form(3000, description="Maximum response length"),
    organization_id: Optional[str] = Form(None, description="Organization ID"),
    user_id: Optional[str] = Form(None, description="User ID"),
    use_hybrid_rag: bool = Form(True, description="Use hybrid RAG for citations")
):
    """
    STANDALONE Chat endpoint - NO prior analysis required
    
    Use this for direct Q&A with the research database without needing 
    to create an initial analysis first.
    
    Supports:
    1. **Text messages** - Questions or instructions
    2. **Document uploads** - PDF, DOCX, DOC, TXT files (content analyzed)
    3. **Image uploads** - PNG, JPG, JPEG, etc. (OCR text extraction)
    4. **Combined inputs** - Any combination of the above
    
    Example with text only:
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat/standalone" \
      -F "message=What research exists on family therapy for youth recidivism?"
    ```
    
    Example with document:
    ```bash
    curl -X POST "http://localhost:8080/single-analysis/chat/standalone" \
      -F "message=Summarize this document and find related research" \
      -F "files[]=@my_document.pdf"
    ```
    
    Response includes:
    - response: Answer with [1], [2] style citations
    - citations: List of research sources used
    - has_research: Whether research citations were found
    """
    
    logger.info("=" * 60)
    logger.info("STANDALONE CHAT ENDPOINT")
    logger.info("=" * 60)
    
    # Extract files from form
    form = await request.form()
    files_raw = form.getlist("files") or form.getlist("files[]")
    
    valid_files = []
    if files_raw:
        for f in files_raw:
            if hasattr(f, 'filename') and hasattr(f, 'read'):
                try:
                    content = await f.read(10)
                    await f.seek(0)
                    if f.filename and f.filename.strip() != "" and len(content) > 0:
                        valid_files.append(f)
                        logger.info(f"  File accepted: {f.filename}")
                except Exception as e:
                    logger.error(f"  Error reading file: {e}")
    
    logger.info(f"Standalone chat - Files: {len(valid_files)}, Message: {len(message)} chars")
    
    if not message.strip() and not valid_files:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'message' or 'files' must be provided"
        )
    
    # Process optional parameters
    actual_domain = None
    if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
        actual_domain = domain.strip()
    
    actual_message = message.strip() if message else "Please analyze the uploaded content."
    
    try:
        # Extract content from files if provided
        from app.services.single_analysis_chat import extract_content_from_upload
        
        uploaded_content = []
        combined_text = actual_message
        
        if valid_files:
            logger.info(f"Extracting content from {len(valid_files)} files...")
            for f in valid_files:
                extraction = await extract_content_from_upload(f)
                if extraction["success"]:
                    uploaded_content.append(extraction)
                    combined_text += f"\n\n[Content from {extraction['filename']}]:\n{extraction['content'][:2000]}"
                    logger.info(f"  ✓ {extraction['filename']}: {extraction['length']} chars")
        
        # Use the hybrid retrieval directly for standalone chat
        import os
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if use_hybrid_rag and index_name:
            try:
                from app.services.hybrid_retrieval import hybrid_rag_pipeline
                
                result = await hybrid_rag_pipeline(
                    query_text=combined_text,
                    index_name=index_name,
                    top_k_retrieval=30,
                    top_n_rerank=top_k,
                    chunk_documents=len(combined_text) > 1500,  # Chunk if long
                    namespace="research",
                    max_tokens=max_tokens
                )
                
                # Add metadata
                result["uploaded_files"] = [c["filename"] for c in uploaded_content]
                result["is_standalone"] = True
                
            except Exception as e:
                logger.warning(f"Hybrid RAG failed: {e}, using fallback")
                from app.services.single_analysis_rag import generate_with_rag_citations
                
                result = await generate_with_rag_citations(
                    system_prompt="You are a research assistant. Answer the user's question using evidence from the research database. Use [1], [2] style citations.",
                    user_query=combined_text,
                    top_k_research=top_k,
                    domain=actual_domain,
                    max_tokens=max_tokens,
                    enable_web_fallback=False
                )
                result["uploaded_files"] = [c["filename"] for c in uploaded_content]
                result["is_standalone"] = True
        else:
            # Fallback without hybrid
            from app.services.single_analysis_rag import generate_with_rag_citations
            
            result = await generate_with_rag_citations(
                system_prompt="You are a research assistant. Answer the user's question using evidence from the research database. Use [1], [2] style citations.",
                user_query=combined_text,
                top_k_research=top_k,
                domain=actual_domain,
                max_tokens=max_tokens,
                enable_web_fallback=False
            )
            result["uploaded_files"] = [c["filename"] for c in uploaded_content]
            result["is_standalone"] = True
        
        logger.info(f"Standalone response: {len(result.get('response', ''))} chars, {result.get('num_sources', 0)} citations")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Standalone chat failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/test-rag-search")
async def test_rag_search_directly(
    query: str = Form("juvenile justice diversion"),
    domain: Optional[str] = Form(None),
    top_k: int = Form(10)
):
    """Test RAG search with any query"""
    from app.services.research import search_research_chunks_from_text
    
   
    actual_domain = None
    if domain and domain.strip() and domain.lower() not in ["none", "null", "string"]:
        actual_domain = domain.strip()
    
    logger.info(f"Test RAG search: '{query}', domain: {repr(actual_domain)}, top_k: {top_k}")
    
    try:
        chunks = await search_research_chunks_from_text(
            query_text=query,
            top_k=top_k,
            domain=actual_domain
        )
        
        return {
            "query": query,
            "domain": actual_domain,
            "top_k": top_k,
            "chunks_found": len(chunks),
            "sample_results": [
                {
                    "filename": chunk.get('filename'),
                    "domain": chunk.get('domain'), 
                    "section": chunk.get('section'),
                    "distance": float(chunk.get('distance', 0)),
                    "content_preview": chunk.get('content', '')[:200] + "..."
                }
                for chunk in chunks[:5]
            ]
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/debug/db-check")
async def debug_db_check():
    """Debug endpoint to check database connections"""
    from app.database import check_database_health
    
    health = check_database_health()
    
    return {
        "database_health": {
            "vector_db": health["vector_db"],
            "django_db": health["django_db"],
            "irs_db": health["irs_db"]
        },
        "django_tables": health.get("django_db_tables", []),
        "vector_tables": health.get("vector_db_tables", []),
        "status": "healthy" if health["django_db"] else "unhealthy"
    }


class StoreAnalysisRequest(BaseModel):
    """Model for storing analysis from Django"""
    analysis_id: str
    response_text: str
    citations: List[Dict[str, Any]] = []
    file_metadata: List[Dict[str, Any]] = []
    content_analysis: Dict[str, Any] = {}
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    chat_type: str = "ANALYSIS"

@router.post("/store-analysis")
async def store_analysis_from_django(request: StoreAnalysisRequest):
    """
    Store analysis from Django into FastAPI vector database
    
    This endpoint is called by Django after creating an analysis
    to ensure it's available in FastAPI for chat functionality.
    """
    logger.info(f"Store analysis request from Django - ID: {request.analysis_id}, type: {request.chat_type}")
    logger.debug(f"Response length: {len(request.response_text)} chars, citations: {len(request.citations)}, org_id: {request.organization_id}, user_id: {request.user_id}")
    
    try:
        from app.database import insert_single_analysis
        
        success = insert_single_analysis(
            analysis_id=request.analysis_id,
            chat_type=request.chat_type,
            response_text=request.response_text,
            citations=request.citations,
            file_metadata=request.file_metadata,
            content_analysis=request.content_analysis,
            organization_id=request.organization_id,
            user_id=request.user_id
        )
        
        if success:
            logger.info(f"Analysis stored successfully in vector DB: {request.analysis_id}")
            return {
                "status": "success", 
                "message": "Analysis stored",
                "analysis_id": request.analysis_id
            }
        else:
            logger.error(f"Failed to store analysis in vector DB: {request.analysis_id}")
            return {
                "status": "error", 
                "message": "Storage failed",
                "analysis_id": request.analysis_id
            }, 500
            
    except Exception as e:
        logger.error(f"Error storing analysis: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": str(e),
            "analysis_id": request.analysis_id
        }, 500

    
__all__ = ["router"]