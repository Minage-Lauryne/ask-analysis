from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import logging
from app.services.comparative_chat import generate_chat_response_from_data
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

DJANGO_URL = os.getenv('DJANGO_URL') or os.getenv('DJANGO_API_URL')

if not DJANGO_URL:
    raise ValueError("DJANGO_URL or DJANGO_API_URL environment variable must be set")

DJANGO_URL = DJANGO_URL.rstrip('/')
if DJANGO_URL.endswith('/api'):
    DJANGO_URL = DJANGO_URL[:-4]

logger.info(f"Django URL configured: {DJANGO_URL}")

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

if not DJANGO_URL:
    logger.warning("Using hardcoded Django URL - environment variable not set")

DJANGO_URL = DJANGO_URL.rstrip('/')

logger.info(f"Django URL finalized: {DJANGO_URL}")

class ChatRequest(BaseModel):
    rfp_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    context_used: Optional[Dict[str, Any]] = None

@router.post("/generate", response_model=ChatResponse)
async def generate_chat_response(chat_request: ChatRequest):
    """
    Generate AI response to user's question based on RFP analysis
    This is called by Django backend
    """
    try:
        logger.info(f"Chat request received - rfp_id: {chat_request.rfp_id}")
        
        internal_url = f"{DJANGO_URL}/api/internal/submissions/{chat_request.rfp_id}/"
        logger.debug(f"Fetching from: {internal_url}")
        
        django_response = session.get(
            internal_url,
            timeout=300,  
            headers={
                'User-Agent': 'ComplereAgent/1.0',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        logger.info(f"Django API response: {django_response.status_code}")
        
        if django_response.status_code != 200:
            logger.error(f"Django API error {django_response.status_code}: {django_response.text[:500]}")
            
            if django_response.status_code == 404:
                error_msg = "Submission not found. Please check if the submission exists and has been analyzed."
            elif django_response.status_code == 401:
                error_msg = "Authentication error. The internal endpoint may not be configured correctly."
            elif django_response.status_code == 500:
                error_msg = "Django server error. Please check Django logs for details."
            else:
                error_msg = f"Error fetching data from Django backend (Status: {django_response.status_code})"
            
            return ChatResponse(
                answer=error_msg,
                context_used={"error": f"Django returned {django_response.status_code}"}
            )
        
        submission_data = django_response.json()
        
        rfp_text = submission_data.get('instructions') or submission_data.get('rfp_text', '')
        rfp_title = submission_data.get('rfp_title', 'Unknown Analysis')
        rfp_summary = submission_data.get('rfp_summary', {})
        proposals = submission_data.get('proposals', [])
        
        for i, proposal in enumerate(proposals):
            org_name = proposal.get('organization_name', 'Unknown')
            score = proposal.get('overall_alignment_score', 0)
            recommendation = proposal.get('recommendation', 'Consider')
            logger.debug(f"Proposal {i+1}: {org_name} - Score: {score} - {recommendation}")
        
        if not proposals:
            return ChatResponse(
                answer="This submission doesn't have any proposals analyzed yet.",
                context_used={"error": "No proposals found"}
            )
        
        if not rfp_text:
            return ChatResponse(
                answer="This submission doesn't have RFP text or instructions available.",
                context_used={"error": "No RFP text"}
            )
        
        logger.info("Generating AI response...")
        result = generate_chat_response_from_data(
            user_question=chat_request.message,
            rfp_text=rfp_text,
            rfp_title=rfp_title,
            rfp_summary=rfp_summary,
            proposals=proposals
        )
        
        logger.info(f"Generated response: {len(result.get('answer', ''))} characters")
        if result.get('context_used'):
            logger.debug(f"Context used: {result['context_used']}")
        
        return ChatResponse(
            answer=result.get('answer', 'Sorry, I could not generate a response.'),
            context_used=result.get('context_used')
        )
        
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching data from Django after 300 seconds")
        return ChatResponse(
            answer="The request timed out while fetching data. The Django service may be slow or unavailable. Please try again.",
            context_used={"error": "Timeout after 300s"}
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return ChatResponse(
            answer=f"Unable to connect to Django backend at {DJANGO_URL}. Please check if the Django service is running and accessible.",
            context_used={"error": "Connection error", "url": DJANGO_URL}
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return ChatResponse(
            answer="Unable to connect to the backend service. Please check if the Django server is running.",
            context_used={"error": "Network error"}
        )
    except Exception as e:
        logger.error(f"Error in chat generation: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return ChatResponse(
            answer="I encountered an error while processing your question. Please try again.",
            context_used={"error": str(e)}
        )
