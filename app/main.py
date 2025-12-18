from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import comparative_analyze, comparative_chat, single_analysis
from app.services.single_analysis_service import SingleAnalysisService
from app.services.research import close_pool

single_analysis_service = SingleAnalysisService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_pool()


app = FastAPI(
    title="RAG Proposal Analyzer Agent", 
    description="AI agent for analyzing proposals with RAG-powered research citations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], 
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(comparative_analyze.router, prefix="/analyze", tags=["Analysis"])
app.include_router(comparative_chat.router, prefix="/chat", tags=["Chat"])
app.include_router(single_analysis.router, prefix="/single-analysis", tags=["Single Analysis"])

@app.get("/")
def root():
    return {"message": "FastAPI RAG Agent is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {
        "name": "RAG Proposal Analyzer Agent",
        "version": "1.0.0",
        "endpoints": {
            "analysis": "/analyze",
            "chat": "/chat", 
            "single_analysis": "/single-analysis"
        },
        "features": [
            "RFP-based proposal analysis",
            "Instruction-based analysis", 
            "Evidence-only analysis",
            "Research-backed single analysis with citations",
            "File upload single analysis with RAG",
            "Organization verification",
            "Chat with analyzed proposals"
        ]
    }


@app.get("/chat-types/")
async def get_chat_types():
    """Get all available chat types for Next Steps"""
    try:
        return single_analysis_service.get_available_chat_types()
    except Exception as e:
        print(f"Error getting chat types: {e}")
        return []

@app.post("/single-analysis/files/v2")
async def single_analysis_files_v2(
    files: List[UploadFile] = File(...),
    chat_type: str = Form("ANALYSIS"),
    domain: Optional[str] = Form(None),
    top_k: int = Form(10),
    max_tokens: int = Form(4000),
    web_fallback: bool = Form(False),
    initial_analysis_context: Optional[str] = Form(None),
    organization_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    context_mode: str = Form("summary")
):
    """
    Enhanced single analysis with file uploads - supports all specialized analyses with context
    Uses the existing SingleAnalysisService.analyze_files method
    """
    try:
        result = await single_analysis_service.analyze_files(
            files=files,
            chat_type=chat_type,
            domain=domain,
            top_k=top_k,
            max_tokens=max_tokens,
            organization_id=organization_id,
            user_id=user_id,
            enable_web_fallback=web_fallback,
            initial_analysis_context=initial_analysis_context,
            context_mode=context_mode
        )
        return result
    except Exception as e:
        print(f"Error in single analysis v2: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}

@app.post("/single-analysis/prepare")
async def prepare_analysis(
    chat_type: str = Form(...),
    user_query: Optional[str] = Form(None),
    initial_analysis_context: Optional[str] = Form(None),
    context_mode: str = Form("summary"),
    top_k: int = Form(10),
    domain: Optional[str] = Form(None),
    max_tokens: int = Form(4000),
    web_fallback: bool = Form(False)
):
    """
    Prepare an analysis request without files (for testing or UI preview)
    """
    try:
        request_data = single_analysis_service.prepare_analysis_request(
            chat_type=chat_type,
            user_query=user_query,
            file_contents=None,
            initial_context=initial_analysis_context,
            context_mode=context_mode,
            top_k=top_k,
            domain=domain,
            max_tokens=max_tokens,
            web_fallback=web_fallback
        )
        
        preview_data = {
            "chat_type": chat_type,
            "chat_type_info": request_data.get("chat_type_info"),
            "has_initial_context": request_data.get("has_initial_context", False),
            "context_mode": context_mode,
            "requires_rag": request_data.get("chat_type_info", {}).get("requires_rag", True),
            "search_query": request_data.get("search_query", ""),
            "system_prompt_preview": request_data.get("messages", [{}])[0].get("content", "")[:500] + "..." if len(request_data.get("messages", [{}])[0].get("content", "")) > 500 else request_data.get("messages", [{}])[0].get("content", ""),
            "user_message_preview": request_data.get("messages", [{}])[1].get("content", "")[:500] + "..." if len(request_data.get("messages", [{}])[1].get("content", "")) > 500 else request_data.get("messages", [{}])[1].get("content", "")
        }
        
        return preview_data
    except Exception as e:
        print(f"Error preparing analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}

@app.post("/single-analysis/text/v2")
async def single_analysis_text_v2(
    message: str = Form(...),
    chat_type: str = Form("ANALYSIS"),
    domain: Optional[str] = Form(None),
    top_k: int = Form(10),
    max_tokens: int = Form(4000),
    web_fallback: bool = Form(False),
    organization_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Enhanced text analysis with RAG
    """
    try:
        result = await single_analysis_service.analyze_text(
            message=message,
            chat_type=chat_type,
            domain=domain,
            top_k=top_k,
            max_tokens=max_tokens,
            organization_id=organization_id,
            user_id=user_id,
            enable_web_fallback=web_fallback
        )
        return result
    except Exception as e:
        print(f"Error in text analysis v2: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}