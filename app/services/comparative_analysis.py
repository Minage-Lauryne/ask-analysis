import os
import json
import re
import logging
import asyncio
from typing import Dict, Any, Optional, List
from app.services.research import (
    format_research_context,
    search_research_chunks_from_text,
)
from app.services.embeddings import embed_text_batch
import numpy as np
from dotenv import load_dotenv
from fastapi import UploadFile
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from app.services.verification import (
        verify_organization,
        build_verification_context,
    )

    VERIFICATION_AVAILABLE = True
    logger.info("Verification module loaded")
except ImportError:
    logger.warning("Verification module not available")
    VERIFICATION_AVAILABLE = False

try:
    from app.database import insert_rfp
    
    # insert_proposal doesn't exist in database.py - define fallback
    def insert_proposal(*args, **kwargs):
        return None

    logger.info("Database module loaded")
except ImportError as e:
    logger.warning(f"Database module not available: {e}")

    def insert_rfp(*args, **kwargs):
        return None

    def insert_proposal(*args, **kwargs):
        return None


load_dotenv()


AGENT_DB_STORAGE_ENABLED = False

google_genai = None
try:
    import google.generativeai as genai

    GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        google_genai = genai
        logger.info("Google Generative AI initialized for embeddings")
except Exception as e:
    logger.error(f"Could not initialize Google AI: {e}")

anthropic_client = None
try:
    from anthropic import Anthropic

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_API_KEY:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized for text generation")
except Exception as e:
    logger.error(f"Could not initialize Anthropic: {e}")


def embed_text_google(text: str) -> Optional[np.ndarray]:
    """Generate embeddings using Google Generative AI"""
    if not google_genai:
        return None

    try:
        truncated = text[:8000]

        result = google_genai.embed_content(
            model="models/embedding-001",
            content=truncated,
            task_type="retrieval_document",
        )

        if isinstance(result, dict):
            embedding = result.get("embedding")
        else:
            embedding = getattr(result, "embedding", None)

        if embedding:
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)
    except Exception as e:
        logger.error(f"[embed_text_google] Error: {e}", exc_info=True)

    return None


def embed_placeholder(text: str, dim: int = 768) -> np.ndarray:
    """Generate deterministic placeholder embedding"""
    import hashlib

    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)

    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


def embed_text(text: str) -> np.ndarray:
    """
    Generate embeddings with priority:
    1. Google Generative AI (primary)
    2. Deterministic placeholder (fallback)
    """
    emb = embed_text_google(text)
    if emb is not None:
        return emb

    logger.debug("[embed_text] Using placeholder embedding")
    return embed_placeholder(text)


def safe_anthropic_generate(
    system_msg: str, user_msg: str, max_tokens: int = 2048
) -> Optional[str]:
    """Generate text using Anthropic Claude"""
    if not anthropic_client:
        return None

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )

        if response.content:
            return response.content[0].text
    except Exception as e:
        logger.error(f"[safe_anthropic_generate] Error: {e}", exc_info=True)

    return None


def safe_google_generate(
    system_msg: str, user_msg: str, max_tokens: int = 2048
) -> Optional[str]:
    """Generate text using Google Generative AI"""
    if not google_genai:
        return None

    try:
        model = google_genai.GenerativeModel("gemini-2.0-flash-exp")
        full_prompt = f"{system_msg}\n\n{user_msg}"

        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": max_tokens,
            },
        )

        return response.text if hasattr(response, "text") else None
    except Exception as e:
        logger.error(f"[safe_google_generate] Error: {e}", exc_info=True)

    return None


def safe_generate(
    system_msg: str, user_msg: str, max_tokens: int = 2048
) -> Optional[str]:
    """
    Generate text with priority:
    1. Anthropic Claude (primary for generation)
    2. Google Generative AI (fallback)
    """
    result = safe_anthropic_generate(system_msg, user_msg, max_tokens)
    if result:
        return result

    logger.info("[safe_generate] Anthropic failed, trying Google AI...")
    result = safe_google_generate(system_msg, user_msg, max_tokens)
    if result:
        return result

    logger.error("[safe_generate] No model available or all calls failed")
    return None


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from model response"""
    if not response_text:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", response_text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def safe_extract_json(response_text: str) -> Optional[Dict[str, Any]]:
    """More flexible JSON extraction that handles various formats"""
    if not response_text:
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", response_text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_pattern = r'\{[^{}]*"(?:organization_name|recommendation|budget|timeline|overall_alignment_score|alignment|evidence)"[^{}]*\}'
    matches = re.findall(json_pattern, cleaned, re.DOTALL | re.IGNORECASE)

    for match in matches:
        try:
            cleaned_match = re.sub(r",\s*}", "}", match)
            cleaned_match = re.sub(r",\s*]", "]", cleaned_match)
            return json.loads(cleaned_match)
        except json.JSONDecodeError:
            continue

    return parse_flexible_format(cleaned)


def parse_flexible_format(text: str) -> Optional[Dict[str, Any]]:
    """Parse various flexible formats that AI might return"""
    result = {}

    org_match = re.search(r'"organization_name"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if org_match:
        result["organization_name"] = org_match.group(1)

    rec_match = re.search(r'"recommendation"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if rec_match:
        result["recommendation"] = rec_match.group(1)

    score_match = re.search(r'"overall_alignment_score"\s*:\s*(\d+)', text)
    if score_match:
        try:
            result["overall_alignment_score"] = int(score_match.group(1))
        except:
            result["overall_alignment_score"] = 50

    budget_match = re.search(r'"budget"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    if budget_match:
        result["budget"] = budget_match.group(1)

    timeline_match = re.search(r'"timeline"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    if timeline_match:
        result["timeline"] = timeline_match.group(1)

    alignment_match = re.search(r'"alignment"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if alignment_match:
        try:
            result["alignment"] = json.loads(alignment_match.group(1))
        except:
            result["alignment"] = {}

    evidence_match = re.search(r'"evidence"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if evidence_match:
        try:
            result["evidence"] = json.loads(evidence_match.group(1))
        except:
            result["evidence"] = {}

    return result if result else None


def build_flexible_analysis_prompt(
    instructions: str, research_context: str = ""
) -> str:
    """Build a flexible prompt that works with any instruction format and includes research sources"""

    research_block = ""

    if research_context:
        research_block = f"""
RESEARCH EVIDENCE (from embedded research corpus):
{research_context}

Each item above may include: chunk_id, paper_id, filename, section, domain, and a content excerpt.
Use these as your primary evidence base when justifying WHAT/HOW/WHO and when filling evidence_sources.
"""

    base_system_prompt = """You are an expert proposal analyst. Analyze the proposal based on the user's instructions and return ONLY valid JSON.

CRITICAL: Return ONLY JSON, no additional text, no markdown formatting."""

    user_prompt = f"""
USER ANALYSIS INSTRUCTIONS:
{instructions}

CENTRAL GOAL:
Evaluate each proposal primarily on how well it aligns with evidence-based practices and sound organizational fundamentals.
Treat any RFP-like text or funder preferences mentioned in the instructions as helpful context, but NOT as the main anchor.
The core job is to compare programs to each other based on what evidence says is impactful.
{research_block}

In your analysis, you MUST explicitly consider:

1) PROGRAM DELIVERY MODEL (WHAT)
2) IMPLEMENTATION & OPERATIONS (HOW)
3) FINANCIAL HEALTH & VIABILITY
4) LEADERSHIP, GOVERNANCE & ORGANIZATIONAL CAPACITY (WHO)

IMPORTANT BEHAVIOUR:
- Do NOT rely on exact keyword matching. Infer meaning from the overall context and natural language.
- If instructions are vague or not perfectly articulated, still perform a full, reasonable analysis using the program text.
- Use the RESEARCH EVIDENCE above as your main source for evidence-backed reasoning.

Return ONLY a JSON object with this EXACT structure:

{{
  "organization_name": "extracted organization name",
  "recommendation": "Recommended|Consider|Not Recommended",
  "budget": "extracted budget or 'Not specified'", 
  "timeline": "extracted timeline or 'Not specified'",
  "overall_alignment_score": 0-100,
  "alignment": {{
    "what_text": "2-4 sentence analysis of the program delivery model: what it offers, who it serves, intended outcomes, and fit with evidence-based practice",
    "what_aligned": true/false,
    "how_text": "2-4 sentence analysis of implementation approach, feasibility, and monitoring/learning plan", 
    "how_aligned": true/false,
    "who_text": "2-4 sentence analysis of organizational capacity, leadership, governance, and stability (use any verification data provided)",
    "who_aligned": true/false,
    "why_text": "2-4 sentence justification for the overall recommendation, tying together WHAT, HOW, WHO, financial realism, and any funder priorities in the instructions"
  }},
  
    "evidence": {{
    "what": ["short supporting quote(s) from the proposal text"],
    "how": ["short supporting quote(s) from the proposal text"],
    "who": ["short supporting quote(s) from the proposal or verification data"]
  }},
   "evidence_sources": {{
    "what": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from the research excerpt used",
        "relevance": "1-2 sentences on how this research supports WHAT"
      }}
    ],
      "how": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from the research excerpt used",
        "relevance": "1-2 sentences on how this research supports HOW"
      }}
    ],
    "who": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from the research excerpt used",
        "relevance": "1-2 sentences on how this research supports WHO or organizational capacity"
      }}
    ]
  }}
}}


SCORING GUIDE:
- 90-100: Excellent alignment with evidence-based practice and strong fundamentals (WHAT/HOW/WHO/finances)
- 80-89: Strong alignment with some minor gaps
- 70-79: Moderate alignment with notable weaknesses or risks
- 50-69: Weak alignment or significant concerns
- Below 50: Poor alignment and/or major risks


Return ONLY the JSON object. No other text."""

    return user_prompt


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks if chunks else [text]


async def extract_text_from_upload(upload: UploadFile) -> str:
    """Extract text from uploaded file (txt, pdf, or docx)"""
    import pdfplumber
    from docx import Document

    content = await upload.read()
    name = upload.filename
    ext = os.path.splitext(name)[1].lower()

    if ext == ".txt":
        return content.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        try:
            temp_path = f"/tmp/temp_upload_{uuid.uuid4()}.pdf"
            with open(temp_path, "wb") as f:
                f.write(content)
            with pdfplumber.open(temp_path) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            os.remove(temp_path)
            return text
        except Exception as e:
            logger.error(f"[extract_text_from_upload] PDF extraction failed: {e}", exc_info=True)
            return ""
    elif ext in [".docx", ".doc"]:
        try:
            temp_path = f"/tmp/temp_upload_{uuid.uuid4()}.docx"
            with open(temp_path, "wb") as f:
                f.write(content)
            doc = Document(temp_path)
            text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
            os.remove(temp_path)
            return text
        except Exception as e:
            logger.error(f"[extract_text_from_upload] DOCX extraction failed: {e}", exc_info=True)
            return ""
    else:
        return content.decode("utf-8", errors="ignore")


def analyze_rfp_document(rfp_text: str) -> Dict[str, Any]:
    """Analyze the RFP document to extract key information"""
    system_msg = "You are an expert RFP analyst. Extract key information from RFP documents in structured JSON format."

    user_msg = f"""Analyze this RFP document and extract the following information:

RFP TEXT:
{rfp_text[:3000]}

Provide a JSON response with this exact structure:
{{
  "title": "A clear, concise title for this RFP (max 100 characters)",
  "executive_summary": "2-3 sentence overview of the RFP purpose and goals",
  "key_requirements": ["requirement 1", "requirement 2", "requirement 3"],
  "target_population": "who the grant/project serves or 'Not specified'",
  "budget_range": "expected budget range if mentioned or 'Not specified'",
  "timeline_expectations": "expected project duration or 'Not specified'",
  "evaluation_criteria": ["criterion 1", "criterion 2", "criterion 3"]
}}

Return ONLY valid JSON, no additional text or markdown."""

    raw = safe_generate(system_msg, user_msg)
    if not raw:
        return {
            "title": "Untitled RFP",
            "executive_summary": "Unable to extract RFP summary",
            "key_requirements": [],
            "target_population": "Not specified",
            "budget_range": "Not specified",
            "timeline_expectations": "Not specified",
            "evaluation_criteria": [],
        }

    parsed = extract_json_from_response(raw)
    if parsed and isinstance(parsed, dict):
        defaults = {
            "title": "Untitled RFP",
            "executive_summary": "Unable to extract summary",
            "key_requirements": [],
            "target_population": "Not specified",
            "budget_range": "Not specified",
            "timeline_expectations": "Not specified",
            "evaluation_criteria": [],
        }
        for key, default_val in defaults.items():
            if key not in parsed:
                parsed[key] = default_val
        return parsed

    return {
        "title": "Untitled RFP",
        "executive_summary": raw[:500] if raw else "Unable to extract summary",
        "key_requirements": [],
        "target_population": "Not specified",
        "budget_range": "Not specified",
        "timeline_expectations": "Not specified",
        "evaluation_criteria": [],
    }


def analyze_single_proposal(
    proposal_name: str,
    proposal_chunks: Dict[str, List[str]],
    proposal_vectors: Dict[str, np.ndarray],
    rfp_text: str,
    rfp_embedding: np.ndarray,
    verification_data: Dict[str, Any],
    top_k_chunks: int = 5,
    research_context: str = "",
) -> Optional[Dict[str, Any]]:
    """Analyze a single proposal against the RFP + evidence + verification."""
    try:
        chunks = proposal_chunks[proposal_name]
        vecs = proposal_vectors[proposal_name]

        if isinstance(vecs, list):
            vecs = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, rfp_embedding.shape[0]), dtype=np.float32)
            )

        scores = (
            (vecs @ rfp_embedding).astype(float)
            if vecs.size
            else np.zeros((len(chunks),), dtype=float)
        )
        top_idxs = (
            np.argsort(scores)[-top_k_chunks:][::-1]
            if len(scores) > 0
            else list(range(min(top_k_chunks, len(chunks))))
        )

        excerpt_texts = []
        for i in top_idxs:
            if 0 <= i < len(chunks):
                chunk = chunks[i]
                text_content = (
                    " ".join([str(x) for x in chunk])
                    if isinstance(chunk, list)
                    else str(chunk)
                )
                excerpt_texts.append(text_content[:1500])

        top_excerpts = "\n\n---EXCERPT---\n\n".join(excerpt_texts).strip()
        verification_context = (
            build_verification_context(verification_data)
            if VERIFICATION_AVAILABLE
            else ""
        )

        system_msg = """
        You are an expert grants and evidence analyst.
Your primary job is to evaluate programmes against evidence-based practice
and organizational fundamentals, NOT just match text to an RFP.

Focus on:
- WHAT: programme delivery model, target group, and intended outcomes
- HOW: implementation approach and operational feasibility
- FINANCES: budget realism and financial viability
- WHO: leadership, governance, and organizational capacity + risk

RFP requirements and funder priorities are IMPORTANT but SECONDARY:
they refine and focus the analysis, but they must NOT override evidence
or fundamental programme quality.

Return ONLY valid JSON with no markdown formatting."""

        research_block = ""
        if research_context:
            research_block = f"""

RESEARCH EVIDENCE (from embedded research corpus):
{research_context}

Each excerpt may include: chunk_id, paper_id, filename, section, domain, and a content sample.
Use these research chunks as your PRIMARY evidence base when judging WHAT/HOW/WHO alignment.
        
        """

        user_msg = f"""PROPOSAL ANALYSIS REQUEST

Organization: {proposal_name}

CENTRAL ANALYSIS GOAL:
Compare this proposal to good practice and the research evidence above,
focusing on:
- Programme delivery model (WHAT)
- Implementation & feasibility (HOW)
- Financial realism and sustainability
- Leadership, governance and organizational capacity (WHO)

Treat the RFP text as SECONDARY guidance to align with funder priorities,
but DO NOT let the RFP override your assessment of programme quality, risk,
or evidence alignment.
{research_block}



RFP CONTEXT (optional, secondary guidance):
{rfp_text[:3000]}

Proposal Excerpts (most relevant sections):
{top_excerpts}

{verification_context}

Analyze this proposal and return ONLY a valid JSON object with this EXACT structure:

{{
  "organization_name": "{proposal_name}",
  "recommendation": "Recommended|Consider|Not Recommended",
  "budget": "extracted budget amount (e.g., '$650,000') or 'Not specified'",
  "timeline": "extracted timeline (e.g., '5 months') or 'Not specified'",
  "overall_alignment_score": 0-100,

  "alignment": {{
    "what_text": "2-4 sentences: WHAT does this programme do, who does it serve, what outcomes does it aim for, and how well does this align with evidence and, secondarily, RFP goals?",
    "what_aligned": true or false,

    "how_text": "2-4 sentences: HOW will this be implemented (activities, partners, intensity, MEL)? Is the approach coherent and feasible given the context and evidence?",
    "how_aligned": true or false,

    "who_text": "2-4 sentences: What is the organizational capacity, leadership experience, stability, and risk profile? Use the verification data above. Discuss: (1) IRS verification, (2) financial capacity (revenue/assets), (3) sanctions status, (4) overall risk level, (5) capacity to execute.",
    "who_aligned": true or false,

    "why_text": "2-4 sentences: Synthesize WHAT/HOW/WHO, financial realism, risk, and evidence to explain the final recommendation."
  }},
  
    "evidence": {{
    "what": ["short quote from the proposal supporting WHAT alignment"],
    "how": ["short quote from the proposal supporting HOW alignment"],
    "who": ["short quote from the proposal and/or verification data supporting WHO alignment"]
  }},
  
    "evidence_sources": {{
    "what": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports or challenges WHAT"
      }}
    ],
     "how": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports HOW"
      }}
    ],
        "who": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports WHO or organizational capacity"
      }}
    ]
  }}
}}

SCORING GUIDELINES (with verification and fundamentals):
- 90-100: Exceptional alignment with evidence + strong WHAT/HOW/WHO + solid finances + LOW risk
- 80-89: Strong alignment with minor gaps or MEDIUM risk
- 70-79: Moderate alignment and/or notable weaknesses (usually "Consider")
- 50-69: Weak alignment or significant concerns (often "Not Recommended")
- Below 50: Poor alignment and/or major risks (typically "Not Recommended")

CRITICAL RULES:
1. If risk_level is CRITICAL (sanctions match), recommendation MUST be "Not Recommended" regardless of content quality.
2. If risk_level is HIGH (unverified or serious concerns), factor this heavily — max score typically 75.
3. If risk_level is MEDIUM (uncertain match or weak financials), cap score at 85.
4. Only LOW risk orgs with strong WHAT/HOW/WHO and finances should score 90+.
5. Base WHO analysis heavily on the verification data provided above.
6. Do NOT rely on exact keyword matches from the RFP; reason from the meaning of the proposal text.

IMPORTANT:
Base your analysis ONLY on the provided excerpts, verification data, and research evidence (if present).
Return ONLY the JSON object with no markdown, no extra text.

"""

        raw = safe_generate(system_msg, user_msg, max_tokens=3000)

        if not raw:
            logger.warning(f"[analyze_single_proposal] No response for {proposal_name}")
            return None

        parsed = extract_json_from_response(raw)

        def default_analysis():
            return {
                "organization_name": proposal_name,
                "recommendation": "Consider",
                "budget": "Not specified",
                "timeline": "Not specified",
                "overall_alignment_score": 50,
                "alignment": {
                    "what_text": "Analysis unavailable",
                    "what_aligned": False,
                    "how_text": "Analysis unavailable",
                    "how_aligned": False,
                    "who_text": "Analysis unavailable",
                    "who_aligned": False,
                    "why_text": "Analysis unavailable",
                },
                "evidence": {"what": [], "how": [], "who": []},
                "evidence_sources": {"what": [], "how": [], "who": []},
            }

        validated = default_analysis()

        if isinstance(parsed, dict):
            validated["organization_name"] = str(
                parsed.get("organization_name", proposal_name)
            )
            validated["recommendation"] = str(parsed.get("recommendation", "Consider"))
            validated["budget"] = str(parsed.get("budget", "Not specified"))
            validated["timeline"] = str(parsed.get("timeline", "Not specified"))

            try:
                score = float(parsed.get("overall_alignment_score", 50))
                validated["overall_alignment_score"] = int(max(0, min(100, score)))
            except:
                validated["overall_alignment_score"] = 50

            if "alignment" in parsed and isinstance(parsed["alignment"], dict):
                align = parsed["alignment"]
                for key in ["what_text", "how_text", "who_text", "why_text"]:
                    if key in align:
                        validated["alignment"][key] = str(align[key])[:1000]
                for key in ["what_aligned", "how_aligned", "who_aligned"]:
                    if key in align:
                        validated["alignment"][key] = bool(align[key])

            if "evidence" in parsed and isinstance(parsed["evidence"], dict):
                for key in ["what", "how", "who"]:
                    if key in parsed["evidence"] and isinstance(
                        parsed["evidence"][key], list
                    ):
                        validated["evidence"][key] = [
                            str(x)[:500] for x in parsed["evidence"][key]
                        ]

            if "evidence_sources" in parsed and isinstance(
                parsed["evidence_sources"], dict
            ):
                es = parsed["evidence_sources"]
                validated["evidence_sources"] = {
                    "what": es.get("what", []) or [],
                    "how": es.get("how", []) or [],
                    "who": es.get("who", []) or [],
                }

        validated["verification"] = verification_data

        logger.info(
            f"[analyze_single_proposal] Completed: {proposal_name} - Score: {validated['overall_alignment_score']} - Risk: {verification_data.get('risk_level', 'UNKNOWN')}"
        )
        return validated

    except Exception as e:
        logger.error(f"[analyze_single_proposal] Exception for {proposal_name}: {e}", exc_info=True)
        return None


async def analyze_proposal_with_instructions(
    proposal_name: str,
    proposal_chunks: Dict[str, List[str]],
    proposal_vectors: Dict[str, np.ndarray],
    instructions: str,
    instructions_embedding: np.ndarray,
    verification_data: Dict[str, Any],
    top_k_chunks: int = 5,
) -> Optional[Dict[str, Any]]:
    """Analyze proposal with flexible instruction handling"""
    try:
        chunks = proposal_chunks[proposal_name]
        vecs = proposal_vectors[proposal_name]

        if isinstance(vecs, list):
            vecs = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, instructions_embedding.shape[0]), dtype=np.float32)
            )

        scores = (
            (vecs @ instructions_embedding).astype(float)
            if vecs.size
            else np.zeros((len(chunks),), dtype=float)
        )
        top_idxs = (
            np.argsort(scores)[-top_k_chunks:][::-1]
            if len(scores) > 0
            else list(range(min(top_k_chunks, len(chunks))))
        )

        excerpt_texts = []
        for i in top_idxs:
            if 0 <= i < len(chunks):
                chunk = chunks[i]
                text_content = (
                    " ".join([str(x) for x in chunk])
                    if isinstance(chunk, list)
                    else str(chunk)
                )
                excerpt_texts.append(text_content[:1500])

        top_excerpts = "\n\n---EXCERPT---\n\n".join(excerpt_texts).strip()
        verification_context = (
            build_verification_context(verification_data)
            if VERIFICATION_AVAILABLE
            else ""
        )

        system_msg = """You are an expert proposal analyst. Analyze the proposal based on the user's instructions and return ONLY valid JSON.

CRITICAL: Return ONLY JSON, no additional text, no markdown formatting."""

        evidence_query = instructions

        research_chunks = await search_research_chunks_from_text(
            evidence_query, top_k=10
        )

        research_context_str = format_research_context(research_chunks)

        user_msg = build_flexible_analysis_prompt(
            instructions, research_context=research_context_str
        )

        user_msg = (
            f"{user_msg}\n\nPROPOSAL EXCERPTS (most relevant sections):\n{top_excerpts}"
        )

        if verification_context:
            user_msg = (
                f"{user_msg}\n\nORGANIZATION VERIFICATION DATA:\n{verification_context}"
            )

        raw = safe_generate(system_msg, user_msg, max_tokens=3000)

        if not raw:
            logger.warning(
                f"[analyze_proposal_with_instructions] No response for {proposal_name}"
            )
            return None

        parsed = safe_extract_json(raw)

        if not parsed:
            logger.warning(
                f"[analyze_proposal_with_instructions] Could not parse JSON for {proposal_name}"
            )
            logger.debug(f"[DEBUG] Raw response: {raw[:500]}...")
            return None

        result = {
            "organization_name": parsed.get("organization_name", proposal_name),
            "recommendation": parsed.get("recommendation", "Consider"),
            "budget": parsed.get("budget", "Not specified"),
            "timeline": parsed.get("timeline", "Not specified"),
            "overall_alignment_score": min(
                100, max(0, parsed.get("overall_alignment_score", 50))
            ),
            "alignment": {
                "what_text": parsed.get("alignment", {}).get(
                    "what_text", "Analysis not available"
                ),
                "what_aligned": parsed.get("alignment", {}).get("what_aligned", False),
                "how_text": parsed.get("alignment", {}).get(
                    "how_text", "Analysis not available"
                ),
                "how_aligned": parsed.get("alignment", {}).get("how_aligned", False),
                "who_text": parsed.get("alignment", {}).get(
                    "who_text", "Analysis not available"
                ),
                "who_aligned": parsed.get("alignment", {}).get("who_aligned", False),
                "why_text": parsed.get("alignment", {}).get(
                    "why_text", "Analysis not available"
                ),
            },
            "evidence": {
                "what": parsed.get("evidence", {}).get("what", []),
                "how": parsed.get("evidence", {}).get("how", []),
                "who": parsed.get("evidence", {}).get("who", []),
            },
            "evidence_sources": {
                "what": parsed.get("evidence_sources", {}).get("what", []),
                "how": parsed.get("evidence_sources", {}).get("how", []),
                "who": parsed.get("evidence_sources", {}).get("who", []),
            },
            "verification": verification_data,
        }

        logger.info(
            f"[analyze_proposal_with_instructions]  {proposal_name} - Score: {result['overall_alignment_score']}"
        )
        return result

    except Exception as e:
        logger.error(
            f"[analyze_proposal_with_instructions] Exception for {proposal_name}: {e}", exc_info=True
        )
        return None


def generate_final_recommendation(
    all_results: List[Dict[str, Any]], analysis_criteria: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate final comparative analysis and recommendation"""

    if not all_results:
        return {
            "final_recommendation": "No proposals to compare",
            "recommended_proposal": None,
            "comparative_analysis": "No analysis available",
            "key_findings": [],
            "selection_criteria": analysis_criteria,
        }

    proposal_summaries = []
    for proposal in all_results:
        summary = {
            "organization_name": proposal.get("organization_name", "Unknown"),
            "filename": proposal.get("filename", "Unknown"),
            "score": proposal.get("overall_alignment_score", 0),
            "recommendation": proposal.get("recommendation", "Consider"),
            "budget": proposal.get("budget", "Not specified"),
            "timeline": proposal.get("timeline", "Not specified"),
            "risk_level": proposal.get("verification", {}).get("risk_level", "UNKNOWN"),
            "verified": proposal.get("verification", {}).get("verified", False),
            "alignment_analysis": proposal.get("alignment", {}),
            "evidence": proposal.get("evidence", {}),
        }
        proposal_summaries.append(summary)

    system_msg = """You are an expert procurement analyst. Provide a final comparative analysis and clear recommendation based on all proposals analyzed."""

    user_msg = f"""FINAL COMPARATIVE ANALYSIS REQUEST

ANALYSIS CRITERIA:
{json.dumps(analysis_criteria, indent=2)}

PROPOSALS ANALYZED:
{json.dumps(proposal_summaries, indent=2)}

Based on the above analysis, provide a FINAL RECOMMENDATION and comparative analysis.

Return ONLY valid JSON with this EXACT structure:

{{
  "final_recommendation": "2-3 sentence overall conclusion stating which proposal is best and why",
  "recommended_proposal": "filename of the best proposal",
  "recommended_organization": "organization name of the best proposal", 
  "comparative_analysis": "3-5 paragraph detailed comparison covering: strengths/weaknesses of each proposal, risk assessment, value for money, and alignment with requirements",
  "key_findings": [
    "Key finding 1 about the proposals",
    "Key finding 2 about risk assessment", 
    "Key finding 3 about budget/value",
    "Key finding 4 about implementation",
    "Key finding 5 about organizational capacity"
  ],
  "selection_criteria_used": {{
    "primary_factors": ["list", "of", "main", "decision", "factors"],
    "risk_considerations": ["list", "of", "risk", "factors", "considered"],
    "value_assessment": ["list", "of", "value", "considerations"]
  }}
}}

CRITICAL DECISION FACTORS:
1. Prioritize proposals with "Recommended" status and LOW risk
2. Consider budget alignment with requirements
3. Evaluate organizational capacity and verification status
4. Assess implementation timeline feasibility
5. Balance score with risk assessment

Return ONLY the JSON object. No additional text."""

    raw = safe_generate(system_msg, user_msg, max_tokens=3000)

    if not raw:
        best_proposal = max(
            all_results, key=lambda x: x.get("overall_alignment_score", 0)
        )
        return {
            "final_recommendation": f"Based on overall alignment scores, {best_proposal.get('organization_name')} is recommended with a score of {best_proposal.get('overall_alignment_score')}/100.",
            "recommended_proposal": best_proposal.get("filename"),
            "recommended_organization": best_proposal.get("organization_name"),
            "comparative_analysis": "Automatic recommendation based on alignment scores. Manual review recommended for detailed analysis.",
            "key_findings": [
                f"Highest scoring proposal: {best_proposal.get('organization_name')} ({best_proposal.get('overall_alignment_score')}/100)",
                f"Risk level: {best_proposal.get('verification', {}).get('risk_level', 'UNKNOWN')}",
                f"Recommendation: {best_proposal.get('recommendation', 'Consider')}",
            ],
            "selection_criteria_used": {
                "primary_factors": ["alignment_score", "risk_assessment"],
                "risk_considerations": ["verification_status", "risk_level"],
                "value_assessment": ["budget_alignment", "timeline_feasibility"],
            },
        }

    parsed = safe_extract_json(raw)
    if parsed and isinstance(parsed, dict):
        return parsed

    best_proposal = max(all_results, key=lambda x: x.get("overall_alignment_score", 0))
    return {
        "final_recommendation": f"Based on comprehensive analysis, {best_proposal.get('organization_name')} is the recommended proposal.",
        "recommended_proposal": best_proposal.get("filename"),
        "recommended_organization": best_proposal.get("organization_name"),
        "comparative_analysis": "Analysis completed. The recommended proposal showed the strongest overall alignment with requirements.",
        "key_findings": [
            "Proposals evaluated based on alignment scores and risk assessment",
            f"Top proposal: {best_proposal.get('organization_name')} with score {best_proposal.get('overall_alignment_score')}/100",
            f"Risk assessment considered in final recommendation",
        ],
        "selection_criteria_used": analysis_criteria,
    }


def parse_instructions_for_summary(instructions: str) -> Dict[str, Any]:
    """Parse instructions to extract structured requirements"""
    system_msg = """You are an expert at parsing analysis instructions. Extract structured requirements from user instructions."""

    user_msg = f"""Parse these analysis instructions and extract key requirements:

INSTRUCTIONS:
{instructions}

Extract the following information and return ONLY valid JSON:

{{
  "title": "A descriptive title for this analysis",
  "executive_summary": "2-3 sentence summary of what to analyze",
  "key_requirements": ["list", "of", "key", "requirements", "mentioned"],
  "target_population": "who this serves or 'Not specified'", 
  "budget_range": "extracted budget range or 'Not specified'",
  "timeline_expectations": "extracted timeline or 'Not specified'",
  "evaluation_criteria": ["list", "of", "evaluation", "criteria", "mentioned"]
}}

Examples:
- If instructions say "budget under $200K", budget_range should be "Under $200,000"
- If instructions say "3 years timeline", timeline_expectations should be "3 years"
- If instructions mention specific populations, include them in target_population

Return ONLY JSON, no additional text."""

    raw = safe_generate(system_msg, user_msg)
    if not raw:
        return {
            "title": f"Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "executive_summary": instructions[:500],
            "key_requirements": ["Custom instruction-based analysis"],
            "target_population": "As specified in instructions",
            "budget_range": "Not specified",
            "timeline_expectations": "Not specified",
            "evaluation_criteria": ["Based on provided instructions"],
        }

    parsed = extract_json_from_response(raw)
    if parsed and isinstance(parsed, dict):
        defaults = {
            "title": f"Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "executive_summary": instructions[:500],
            "key_requirements": ["Custom instruction-based analysis"],
            "target_population": "As specified in instructions",
            "budget_range": "Not specified",
            "timeline_expectations": "Not specified",
            "evaluation_criteria": ["Based on provided instructions"],
        }
        for key, default_val in defaults.items():
            if key not in parsed:
                parsed[key] = default_val
        return parsed

    return {
        "title": f"Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "executive_summary": instructions[:500],
        "key_requirements": ["Custom instruction-based analysis"],
        "target_population": "As specified in instructions",
        "budget_range": "Not specified",
        "timeline_expectations": "Not specified",
        "evaluation_criteria": ["Based on provided instructions"],
    }


def store_rfp_in_db(
    rfp_text: str,
    rfp_summary: Dict[str, Any],
    filename: str,
    organization_id: Optional[str] = None,
    user_id: str = None,
) -> str:
    """Store RFP in agent.rfps table and return the ID"""
    rfp_id = str(uuid.uuid4())
    if not AGENT_DB_STORAGE_ENABLED:
        logger.info("[store_rfp_in_db] SKIPPED - Django handles storage")
        return rfp_id
    return rfp_id


def store_proposal_in_db(
    rfp_id: str,
    proposal_data: Dict[str, Any],
    filename: str,
    full_text: str,
    organization_id: Optional[str] = None,
    user_id: str = None,
) -> str:
    """Store proposal analysis in agent.proposals table and return the ID"""
    proposal_id = str(uuid.uuid4())

    if not AGENT_DB_STORAGE_ENABLED:
        logger.info("[store_proposal_in_db] SKIPPED - Django handles storage")
        return proposal_id

    return proposal_id


async def analyze_uploaded_files(
    rfp_file: UploadFile,
    proposal_files: List[UploadFile],
    organization_id: Optional[str] = None,
    user_id: str = None,
    submission_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function: Analyze RFP and proposals with verification
    Returns analysis data (Django stores everything)
    """
    logger.info("=" * 60)
    logger.info("STARTING RFP-BASED ANALYSIS WITH VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Submission ID from Django: {submission_id}")

    try:
        analysis_id = submission_id if submission_id else str(uuid.uuid4())

        rfp_text = await extract_text_from_upload(rfp_file)

        rfp_emb = embed_text(rfp_text)

        rfp_summary = analyze_rfp_document(rfp_text)

        evidence_query_parts = [
            rfp_summary.get("title", ""),
            rfp_summary.get("executive_summary", ""),
            " | ".join(rfp_summary.get("key_requirements", [])),
            rfp_summary.get("target_population", ""),
        ]

        evidence_query_text = (
            "\n".join(p for p in evidence_query_parts if p) or rfp_text[:2000]
        )

        try:
            research_chunks = await search_research_chunks_from_text(
                query_text=evidence_query_text,
                top_k=10, 
                domain=None,  
            )
            research_context_str = format_research_context(
                research_chunks, max_chars=600  
            )
        except Exception as e:
            logger.error(f" Research evidence retrieval failed: {e}")
            research_chunks = []
            research_context_str = ""

        proposal_chunks: Dict[str, List[str]] = {}
        proposal_vectors: Dict[str, np.ndarray] = {}
        proposal_texts: Dict[str, str] = {}
        verification_results: Dict[str, Dict[str, Any]] = {}

        for i, pf in enumerate(proposal_files, 1):
            name = pf.filename

            text = await extract_text_from_upload(pf)
            proposal_texts[name] = text

            if VERIFICATION_AVAILABLE:
                logger.info(f" Verifying organization...")
                loop = asyncio.get_event_loop()
                verification_results[name] = await loop.run_in_executor(None, verify_organization, text)

            else:
                verification_results[name] = {
                    "org_name": "Unknown",
                    "verified": False,
                    "risk_level": "UNKNOWN",
                    "ein": None,
                    "revenue": 0,
                    "assets": 0,
                }

            chunks = chunk_text(text)
            logger.info(f"Split into {len(chunks)} chunks")

            vecs = await embed_text_batch(chunks, batch_size=10)
            mat = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, rfp_emb.shape[0]), dtype=np.float32)
            )

            proposal_chunks[name] = chunks
            proposal_vectors[name] = mat
            logger.info(" Generated embeddings")

        logger.info("[4/4] Analyzing proposals against RFP...")
        all_results: List[Dict[str, Any]] = []
        proposal_names = list(proposal_chunks.keys())

        for i, pname in enumerate(proposal_names, 1):
            res = analyze_single_proposal(
                pname,
                proposal_chunks,
                proposal_vectors,
                rfp_text,
                rfp_emb,
                verification_results[pname],
                research_context=research_context_str,
            )
            if res:

                try:
                    proposal_text_for_evidence = proposal_texts.get(pname, "")[:4000]

                    evidence_alignment = await analyze_proposal_evidence_alignment(
                        proposal_name=pname,
                        proposal_text=proposal_text_for_evidence,
                        instructions=rfp_text,  # optional: use RFP text as context
                        top_k_research=10,
                        domain=None,  # or "education"/"health" if you start tagging
                    )

                    res["evidence_alignment"] = evidence_alignment
                except Exception as e:
                    logger.error(f"Evidence-only analysis failed for {pname}: {e}")

            if res:
                res["filename"] = pname
                if research_chunks:
                    res["research_chunks_used"] = research_chunks
                all_results.append(res)
            else:
                logger.warning(" Analysis failed")

        all_results.sort(
            key=lambda x: x.get("overall_alignment_score", 0), reverse=True
        )

        final_recommendation = generate_final_recommendation(all_results, rfp_summary)
        if all_results:
            logger.info(
                f"  Recommended: {final_recommendation.get('recommended_organization', 'None')}"
            )
        else:
            logger.info("  No proposals analyzed")

        logger.info("RFP-BASED ANALYSIS COMPLETE")
        if all_results:
            logger.info(
                f"Final recommendation: {final_recommendation.get('recommended_organization', 'None')}"
            )

        return {
            "rfp_id": analysis_id,
            "rfp_title": rfp_summary.get("title", rfp_file.filename),
            "rfp_summary": rfp_summary,
            "proposals": all_results,
            "final_recommendation": final_recommendation,
            "analysis_type": "rfp_based",
        }

    except Exception as e:
        logger.error(f"RFP-BASED ANALYSIS FAILED: {e}")
        import traceback

        traceback.print_exc()

        error_id = submission_id if submission_id else str(uuid.uuid4())
        return {
            "rfp_id": error_id,
            "rfp_title": "Analysis Failed",
            "rfp_summary": {
                "title": "Error",
                "executive_summary": f"Analysis failed: {str(e)}",
                "key_requirements": [],
                "target_population": "Not specified",
                "budget_range": "Not specified",
                "timeline_expectations": "Not specified",
                "evaluation_criteria": [],
            },
            "proposals": [],
            "final_recommendation": {
                "final_recommendation": "Analysis failed - unable to provide recommendation",
                "recommended_proposal": None,
                "comparative_analysis": "Analysis process encountered an error",
                "key_findings": ["Analysis failed due to technical error"],
            },
            "analysis_type": "rfp_based",
        }


async def analyze_with_instructions(
    instructions: str,
    proposal_files: List[UploadFile],
    organization_id: Optional[str] = None,
    user_id: str = None,
    submission_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze proposals based on custom instructions (no RFP required)
    Returns analysis data for Django to store
    """
    logger.info("STARTING INSTRUCTION-BASED COMPARATIVE ANALYSIS")
    logger.info(f"Submission ID from Django: {submission_id}")

    try:
        analysis_id = submission_id if submission_id else str(uuid.uuid4())
        instructions_emb = embed_text(instructions)

        rfp_summary = parse_instructions_for_summary(instructions)
        logger.info(
            f"    - Timeline: {rfp_summary.get('timeline_expectations', 'Not specified')}"
        )
        logger.info(
            f"    - Key requirements: {len(rfp_summary.get('key_requirements', []))} items"
        )

        proposal_chunks: Dict[str, List[str]] = {}
        proposal_vectors: Dict[str, np.ndarray] = {}
        proposal_texts: Dict[str, str] = {}
        verification_results: Dict[str, Dict[str, Any]] = {}

        for i, pf in enumerate(proposal_files, 1):
            name = pf.filename

            text = await extract_text_from_upload(pf)
            proposal_texts[name] = text

            if VERIFICATION_AVAILABLE:
                logger.info(" Verifying organization...")
                loop = asyncio.get_event_loop()
                verification_results[name] = await loop.run_in_executor(None, verify_organization, text)

            else:
                verification_results[name] = {
                    "org_name": "Unknown",
                    "verified": False,
                    "risk_level": "UNKNOWN",
                    "ein": None,
                    "revenue": 0,
                    "assets": 0,
                }

            chunks = chunk_text(text)

            vecs = await embed_text_batch(chunks, batch_size=10)
            mat = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, instructions_emb.shape[0]), dtype=np.float32)
            )

            proposal_chunks[name] = chunks
            proposal_vectors[name] = mat

        all_results: List[Dict[str, Any]] = []
        proposal_names = list(proposal_chunks.keys())

        for i, pname in enumerate(proposal_names, 1):
            res = await analyze_proposal_with_instructions(
                pname,
                proposal_chunks,
                proposal_vectors,
                instructions,
                instructions_emb,
                verification_results[pname],
            )

            if res:
                evidence_alignment = await analyze_proposal_evidence_alignment(
                    proposal_name=pname,
                    proposal_text="\n\n".join(proposal_chunks[pname])[:4000],
                    instructions=instructions,
                    top_k_research=10,
                    domain=None,
                )

                res["evidence_alignment"] = evidence_alignment
                res["filename"] = pname
                all_results.append(res)

            else:
                logger.warning(f" Analysis failed")

        all_results.sort(
            key=lambda x: x.get("overall_alignment_score", 0), reverse=True
        )

        logger.info(f"\n[5/5] Generating final comparative analysis...")
        final_recommendation = generate_final_recommendation(all_results, rfp_summary)

        logger.info(" INSTRUCTION-BASED ANALYSIS COMPLETE")

        return {
            "rfp_id": analysis_id,
            "rfp_title": rfp_summary["title"],
            "rfp_summary": rfp_summary,
            "proposals": all_results,
            "final_recommendation": final_recommendation,
            "analysis_type": "instruction_based",
        }

    except Exception as e:
        logger.error(f"Instruction-based analysis failed: {e}")
        import traceback

        traceback.print_exc()

        error_id = submission_id if submission_id else str(uuid.uuid4())
        return {
            "rfp_id": error_id,
            "rfp_title": "Analysis Failed",
            "rfp_summary": {
                "title": "Error",
                "executive_summary": f"Analysis failed: {str(e)}",
                "key_requirements": [],
                "target_population": "Not specified",
                "budget_range": "Not specified",
                "timeline_expectations": "Not specified",
                "evaluation_criteria": [],
            },
            "proposals": [],
            "final_recommendation": {
                "final_recommendation": "Analysis failed - unable to provide recommendation",
                "recommended_proposal": None,
                "comparative_analysis": "Analysis process encountered an error",
                "key_findings": ["Analysis failed due to technical error"],
            },
            "analysis_type": "instruction_based",
        }


async def analyze_proposal_evidence_alignment(
    proposal_name: str,
    proposal_text: str,
    instructions: Optional[str] = None,
    domain: Optional[str] = None,
    top_k_research: int = 10,
) -> Dict[str, Any]:
    """
    Evidence-only analysis:
    - Uses the proposal text (and optional instructions) to search the research vector DB
    - Builds a research context
    - Asks the model to compare the programme design against the evidence
    - Returns structured alignment + evidence_sources for WHAT / HOW / WHO
    """

    query_parts = [proposal_text[:2000]]
    if instructions:
        query_parts.append(instructions[:1000])
    evidence_query_text = "\n".join(query_parts)

    try:
        research_chunks = await search_research_chunks_from_text(
            query_text=evidence_query_text,
            top_k=top_k_research,
            domain=domain,
        )
        research_context_str = format_research_context(
            research_chunks,
            max_chars=600,
        )
    except Exception as e:
        logger.error(f"[analyze_proposal_evidence_alignment] Evidence retrieval failed: {e}")
        research_chunks = []
        research_context_str = ""

    system_msg = """You are an expert evidence-synthesis analyst.
Your sole job in this step is to compare the programme described
in the proposal against the research evidence provided.

Ignore funder-specific RFPs, procurement rules, or organizational risk.
Focus ONLY on: how well does the programme design match what the evidence suggests works?

Return ONLY valid JSON with no markdown formatting."""

    research_block = ""
    if research_context_str:
        research_block = f"""

RESEARCH EVIDENCE (from embedded research corpus):
{research_context_str}

Each excerpt may include: chunk_id, paper_id, filename, section, domain, and a content sample.
Use these research chunks as your PRIMARY anchor when judging WHAT/HOW/WHO alignment.
"""

    user_msg = f"""EVIDENCE-BASED ALIGNMENT ANALYSIS

Organization: {proposal_name}

PROPOSAL (excerpt):
{proposal_text[:3000]}
{research_block}

TASK:
Compare this programme to the research evidence above.
Focus on three dimensions:

1) WHAT ALIGNMENT (Programme model & outcomes)
   - What is the programme trying to do?
   - Who is it serving?
   - What outcomes does it aim for?
   - Does the evidence suggest this kind of programme model is effective?

2) HOW ALIGNMENT (Implementation approach)
   - How is the programme delivered (intensity, modality, frequency, staffing)?
   - Are there critical design elements (e.g. dose, delivery channel) supported by evidence?
   - Does the approach match what has worked in similar contexts?

3) WHO / CONTEXT ALIGNMENT (Population & setting)
   - Are the target populations, contexts, or systems similar to those in the research?
   - Are there important differences that make evidence less transferable?

Return ONLY a JSON object with this EXACT structure:

{{
  "organization_name": "{proposal_name}",
  "evidence_alignment_score": 0-100,
  "summary": "3-5 sentence plain-language summary of how this proposal aligns with the research overall.",

  "what_alignment": {{
    "score": 0-100,
    "summary": "2-3 sentences on alignment of the programme model, target group, and intended outcomes with evidence.",
    "evidence_sources": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports or challenges WHAT"
      }}
    ]
  }},

  "how_alignment": {{
    "score": 0-100,
    "summary": "2-3 sentences on alignment of implementation approach, intensity, and delivery mode with evidence.",
    "evidence_sources": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports HOW"
      }}
    ]
  }},

  "who_alignment": {{
    "score": 0-100,
    "summary": "2-3 sentences on how similar the proposal's target population/context is to the contexts in the evidence.",
    "evidence_sources": [
      {{
        "chunk_id": "chunk_id from research corpus or 'N/A'",
        "paper_id": "paper_id or 'N/A'",
        "filename": "filename or 'N/A'",
        "section": "section or 'N/A'",
        "domain": "domain or 'N/A'",
        "quote": "short quote from research excerpt used",
        "relevance": "1-2 sentences on how this research supports WHO/context alignment"
      }}
    ]
  }}
}}

SCORING GUIDE:
- 90-100: Strongly supported by evidence across most key features
- 80-89: Well aligned with evidence with some gaps or uncertainty
- 70-79: Mixed alignment (some parts supported, others unclear or weak)
- 50-69: Weak alignment or limited supporting evidence
- Below 50: Poor alignment or evidence suggests low effectiveness

Return ONLY the JSON object. No extra text."""

    raw = safe_generate(system_msg, user_msg, max_tokens=2800)
    parsed = safe_extract_json(raw) if raw else None

    def default_alignment() -> Dict[str, Any]:
        return {
            "organization_name": proposal_name,
            "evidence_alignment_score": 50,
            "summary": "Evidence alignment analysis unavailable or incomplete.",
            "what_alignment": {
                "score": 50,
                "summary": "Not available",
                "evidence_sources": [],
            },
            "how_alignment": {
                "score": 50,
                "summary": "Not available",
                "evidence_sources": [],
            },
            "who_alignment": {
                "score": 50,
                "summary": "Not available",
                "evidence_sources": [],
            },
            "research_chunks_used": research_chunks,
        }

    if not parsed or not isinstance(parsed, Dict):
        logger.error(
            f"Failed to parse JSON for {proposal_name}"
        )
        return default_alignment()

    result = default_alignment()

    try:
        result["organization_name"] = str(
            parsed.get("organization_name", proposal_name)
        )

        try:
            overall_score = float(parsed.get("evidence_alignment_score", 50))
            result["evidence_alignment_score"] = int(max(0, min(100, overall_score)))
        except Exception:
            pass

        result["summary"] = str(parsed.get("summary", result["summary"]))[:2000]

        for dim in ["what_alignment", "how_alignment", "who_alignment"]:
            if dim in parsed and isinstance(parsed[dim], dict):
                dim_parsed = parsed[dim]

                try:
                    s = float(dim_parsed.get("score", 50))
                    result[dim]["score"] = int(max(0, min(100, s)))
                except Exception:
                    pass

                if "summary" in dim_parsed:
                    result[dim]["summary"] = str(dim_parsed["summary"])[:2000]

                if "evidence_sources" in dim_parsed and isinstance(
                    dim_parsed["evidence_sources"], list
                ):
                    cleaned_sources = []
                    for src in dim_parsed["evidence_sources"][:10]:
                        if not isinstance(src, dict):
                            continue
                        cleaned_sources.append(
                            {
                                "chunk_id": str(src.get("chunk_id", "N/A")),
                                "paper_id": str(src.get("paper_id", "N/A")),
                                "filename": str(src.get("filename", "N/A")),
                                "section": str(src.get("section", "N/A")),
                                "domain": str(src.get("domain", "N/A")),
                                "quote": str(src.get("quote", ""))[:1000],
                                "relevance": str(src.get("relevance", ""))[:1000],
                            }
                        )
                    result[dim]["evidence_sources"] = cleaned_sources

        result["research_chunks_used"] = research_chunks

    except Exception as e:
        logger.error(
            f"Error normalizing result for {proposal_name}: {e}"
        )

    return result


async def analyze_with_evidence_only(
    proposal_files: List[UploadFile],
    organization_id: Optional[str] = None,
    user_id: str = None,
    submission_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze proposals based purely on evidence-based practice (no RFP or instructions required)
    This mode evaluates proposals against research evidence and organizational fundamentals
    """
    logger.info("STARTING EVIDENCE-ONLY ANALYSIS")

    try:
        analysis_id = submission_id if submission_id else str(uuid.uuid4())
        logger.info(f"Using analysis ID: {analysis_id}")
        default_instructions = """
Evaluate each proposal based on evidence-based practice and organizational fundamentals.

Key evaluation areas:
1. PROGRAM DELIVERY MODEL (WHAT): Does the program model align with research evidence for what works? Are the target population, intended outcomes, and intervention approach supported by evidence?

2. IMPLEMENTATION & OPERATIONS (HOW): Is the implementation approach feasible and supported by evidence? Are activities, partnerships, intensity, and monitoring/evaluation plans realistic?

3. ORGANIZATIONAL CAPACITY (WHO): Does the organization have the leadership, governance, financial stability, and capacity to execute the program effectively?

4. FINANCIAL VIABILITY: Is the budget realistic and sustainable? Does the organization demonstrate sound financial management?

Score proposals primarily on evidence alignment and organizational fundamentals, not on adherence to any specific funder requirements.
"""

        instructions_emb = embed_text(default_instructions)

        rfp_summary = {
            "title": "Evidence-Based Comparative Analysis",
            "executive_summary": "Evaluating proposals based on evidence-based practice and organizational fundamentals without reference to a specific RFP or funder requirements.",
            "key_requirements": [
                "Evidence-based program model",
                "Feasible implementation approach",
                "Strong organizational capacity",
                "Sound financial management",
            ],
            "target_population": "As specified by each proposal",
            "budget_range": "As specified by each proposal",
            "timeline_expectations": "As specified by each proposal",
            "evaluation_criteria": [
                "Alignment with research evidence",
                "Implementation feasibility",
                "Organizational capacity and stability",
                "Financial viability and sustainability",
            ],
        }

        proposal_chunks: Dict[str, List[str]] = {}
        proposal_vectors: Dict[str, np.ndarray] = {}
        proposal_texts: Dict[str, str] = {}
        verification_results: Dict[str, Dict[str, Any]] = {}

        for i, pf in enumerate(proposal_files, 1):
            name = pf.filename

            text = await extract_text_from_upload(pf)
            proposal_texts[name] = text

            if VERIFICATION_AVAILABLE:
                loop = asyncio.get_event_loop()
                verification_results[name] = await loop.run_in_executor(None, verify_organization, text)

            else:
                verification_results[name] = {
                    "org_name": "Unknown",
                    "verified": False,
                    "risk_level": "UNKNOWN",
                    "ein": None,
                    "revenue": 0,
                    "assets": 0,
                }

            chunks = chunk_text(text)

            vecs = await embed_text_batch(chunks, batch_size=10)
            mat = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, instructions_emb.shape[0]), dtype=np.float32)
            )

            proposal_chunks[name] = chunks
            proposal_vectors[name] = mat

        all_results: List[Dict[str, Any]] = []
        proposal_names = list(proposal_chunks.keys())

        for i, pname in enumerate(proposal_names, 1):
            res = await analyze_proposal_with_instructions(
                pname,
                proposal_chunks,
                proposal_vectors,
                default_instructions,
                instructions_emb,
                verification_results[pname],
            )

            if res:
                evidence_alignment = await analyze_proposal_evidence_alignment(
                    proposal_name=pname,
                    proposal_text="\n\n".join(proposal_chunks[pname])[:4000],
                    instructions=default_instructions,
                    top_k_research=10,
                    domain=None,
                )

                res["evidence_alignment"] = evidence_alignment
                res["filename"] = pname
                all_results.append(res)
                logger.info(
                    f" Score: {res.get('overall_alignment_score', 0)}/100 - {res.get('recommendation')}"
                )
            else:
                logger.error(f"Analysis failed")

        all_results.sort(
            key=lambda x: x.get("overall_alignment_score", 0), reverse=True
        )

        final_recommendation = generate_final_recommendation(all_results, rfp_summary)
        logger.info("EVIDENCE-ONLY ANALYSIS COMPLETE")

        return {
            "rfp_id": analysis_id,
            "rfp_title": rfp_summary["title"],
            "rfp_summary": rfp_summary,
            "proposals": all_results,
            "final_recommendation": final_recommendation,
            "analysis_type": "evidence_only",
        }

    except Exception as e:
        logger.error(f"EVIDENCE-ONLY ANALYSIS FAILED: {e}")
        import traceback

        traceback.print_exc()

        error_id = submission_id if submission_id else str(uuid.uuid4())
        return {
            "rfp_id": error_id,
            "rfp_title": "Analysis Failed",
            "rfp_summary": {
                "title": "Error",
                "executive_summary": f"Analysis failed: {str(e)}",
                "key_requirements": [],
                "target_population": "Not specified",
                "budget_range": "Not specified",
                "timeline_expectations": "Not specified",
                "evaluation_criteria": [],
            },
            "proposals": [],
            "final_recommendation": {
                "final_recommendation": "Analysis failed - unable to provide recommendation",
                "recommended_proposal": None,
                "comparative_analysis": "Analysis process encountered an error",
                "key_findings": ["Analysis failed due to technical error"],
            },
            "analysis_type": "evidence_only",
        }
