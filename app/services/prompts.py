"""
Prompt Manager - Centralized prompt management for all analysis types
"""

import logging

from .comparative_analysis import (
    START_ANALYSIS_PROMPT,
    CONTINUE_ANALYSIS_PROMPT,
    OUTPUT_FORMAT_NOTICE,
    get_example_citations
)
from .bias import START_BIAS_PROMPT, CONTINUE_BIAS_PROMPT
from .counterpoint import START_COUNTERPOINT_PROMPT, CONTINUE_COUNTERPOINT_PROMPT
from .landscape import START_LANDSCAPE_PROMPT, CONTINUE_LANDSCAPE_PROMPT
from .summary import START_SUMMARY_PROMPT, CONTINUE_SUMMARY_PROMPT
from .board_memo import BOARD_MEMO_PROMPT
from .comprehensive_report import COMPREHENSIVE_REPORT_PROMPT
from .discussion_questions import DISCUSSION_QUESTIONS_PROMPT
from .financial_analysis import FINANCIAL_ANALYSIS_PROMPT
from .leadership_analysis import LEADERSHIP_ANALYSIS_PROMPT
from .program_analysis import PROGRAM_ANALYSIS_PROMPT
from .relevant_research import RELEVANT_RESEARCH_PROMPT
from .site_visit_prep import SITE_VISIT_PREP_GUIDE_PROMPT
from .strategic_planning import STRATEGIC_PLANNING_GUIDE_PROMPT
from .data_citation import DATA_CITATION_PROMPT
from .narrative import NARRATIVE_PROMPT

logger = logging.getLogger(__name__)

CHAT_TYPES = {
    "ANALYSIS": "General Analysis",
    "BIAS": "Bias Analysis", 
    "COUNTERPOINT": "Counterpoint Analysis",
    "LANDSCAPE_ANALYSIS": "Landscape Analysis",
    "SUMMARY": "Summary",
    "BOARD_MEMO": "Board Memo",
    "COMPREHENSIVE_REPORT": "Comprehensive Report",
    "DISCUSSION_QUESTIONS": "Discussion Questions",
    "FINANCIAL_ANALYSIS": "Financial Analysis",
    "LEADERSHIP_ANALYSIS": "Leadership Analysis",
    "PROGRAM_ANALYSIS": "Program Analysis",
    "RELEVANT_RESEARCH": "Relevant Research",
    "SITE_VISIT_PREP_GUIDE": "Site Visit Prep Guide",
    "STRATEGIC_PLANNING_GUIDE": "Strategic Planning Guide",
    "DATA_CITATION": "Data Citation & Validation",
    "NARRATIVE": "Illustrative Fact-Based Narrative",
}

PROMPT_MAP = {
    "ANALYSIS": {
        "start": START_ANALYSIS_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Comprehensive analysis of documents with research citations",
        "with_context": True,
        "requires_rag": True
    },
    "BIAS": {
        "start": START_BIAS_PROMPT,
        "continue": CONTINUE_BIAS_PROMPT,
        "description": "Identify biases, assumptions, and blind spots in analysis",
        "with_context": True,
        "requires_rag": True
    },
    "COUNTERPOINT": {
        "start": START_COUNTERPOINT_PROMPT,
        "continue": CONTINUE_COUNTERPOINT_PROMPT,
        "description": "Provide alternative perspectives and critical analysis",
        "with_context": True,
        "requires_rag": True
    },
    "LANDSCAPE_ANALYSIS": {
        "start": START_LANDSCAPE_PROMPT,
        "continue": CONTINUE_LANDSCAPE_PROMPT,
        "description": "Analyze competitive landscape and ecosystem positioning",
        "with_context": True,
        "requires_rag": True
    },
    "SUMMARY": {
        "start": START_SUMMARY_PROMPT,
        "continue": CONTINUE_SUMMARY_PROMPT,
        "description": "Concise 2-page executive summary of key findings",
        "with_context": True,
        "requires_rag": False
    },
    "BOARD_MEMO": {
        "start": BOARD_MEMO_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Formal board briefing document with evidence-based analysis",
        "with_context": True,
        "requires_rag": True
    },
    "COMPREHENSIVE_REPORT": {
        "start": COMPREHENSIVE_REPORT_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Detailed comprehensive report with all analysis dimensions",
        "with_context": True,
        "requires_rag": True
    },
    "DISCUSSION_QUESTIONS": {
        "start": DISCUSSION_QUESTIONS_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Generate strategic discussion questions for deep analysis",
        "with_context": True,
        "requires_rag": False
    },
    "FINANCIAL_ANALYSIS": {
        "start": FINANCIAL_ANALYSIS_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Detailed financial analysis with ratios and benchmarks",
        "with_context": True,
        "requires_rag": True
    },
    "LEADERSHIP_ANALYSIS": {
        "start": LEADERSHIP_ANALYSIS_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Analyze leadership capacity, governance, and culture",
        "with_context": True,
        "requires_rag": True
    },
    "PROGRAM_ANALYSIS": {
        "start": PROGRAM_ANALYSIS_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Comprehensive analysis of program effectiveness and impact",
        "with_context": True,
        "requires_rag": True
    },
    "RELEVANT_RESEARCH": {
        "start": RELEVANT_RESEARCH_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Synthesize relevant research and evidence base",
        "with_context": True,
        "requires_rag": True
    },
    "SITE_VISIT_PREP_GUIDE": {
        "start": SITE_VISIT_PREP_GUIDE_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Comprehensive guide for site visit preparation and evaluation",
        "with_context": True,
        "requires_rag": False
    },
    "STRATEGIC_PLANNING_GUIDE": {
        "start": STRATEGIC_PLANNING_GUIDE_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Strategic planning framework and implementation guide",
        "with_context": True,
        "requires_rag": False
    },
    "DATA_CITATION": {
        "start": DATA_CITATION_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Verify data points and provide proper citations",
        "with_context": True,
        "requires_rag": True
    },
    "NARRATIVE": {
        "start": NARRATIVE_PROMPT,
        "continue": CONTINUE_ANALYSIS_PROMPT,
        "description": "Create compelling fact-based narratives from data",
        "with_context": True,
        "requires_rag": False
    },
}

def get_prompt_by_chat_type(
    chat_type: str, 
    is_first_message: bool = True,
    has_initial_context: bool = False,
    context_mode: str = "summary"
) -> str:
    """
    Get the appropriate prompt based on chat type and context
    
    Args:
        chat_type: The type of analysis (e.g., "ANALYSIS", "BIAS", etc.)
        is_first_message: Whether this is the first message in the chat
        has_initial_context: Whether there's initial analysis context available
        context_mode: Either "summary" or "full_documents"
    
    Returns:
        Formatted prompt string
    """
    chat_type = chat_type.upper()
    
    if chat_type not in PROMPT_MAP:
        logger.warning(f"Unknown chat_type '{chat_type}', defaulting to 'ANALYSIS'")
        chat_type = "ANALYSIS"
    
    prompt_info = PROMPT_MAP[chat_type]
    
    if is_first_message:
        base_prompt = prompt_info["start"]
    else:
        base_prompt = prompt_info["continue"]
    
    if has_initial_context and prompt_info.get("with_context", True):
        context_prefix = _build_context_prefix(chat_type, context_mode)
        return context_prefix + "\n\n" + base_prompt
    
    return base_prompt

def _build_context_prefix(chat_type: str, context_mode: str) -> str:
    """Build context prefix for specialized analyses"""
    
    if context_mode == "summary":
        prefix = f"""
## INITIAL ANALYSIS CONTEXT

You are conducting a {CHAT_TYPES[chat_type]} based on a comprehensive initial analysis.

Below is a summary of the initial analysis findings for context. Use this as foundation for your specialized analysis.
Focus on building upon these findings rather than re-analyzing from scratch.

Context will be provided in the user message.
"""
    else:
        prefix = f"""
## DOCUMENT CONTEXT

You are conducting a {CHAT_TYPES[chat_type]} based on the provided documents.

Since no comprehensive initial analysis summary is available, you will work directly with the document contents.

Focus on providing your specialized analysis based on the document evidence.
"""
    
    return prefix

def get_chat_type_info(chat_type: str) -> dict:
    """Get information about a specific chat type"""
    chat_type = chat_type.upper()
    
    if chat_type not in PROMPT_MAP:
        logger.warning(f"Unknown chat_type '{chat_type}' in get_chat_type_info")
        return {
            "name": "Unknown",
            "description": "Unknown analysis type",
            "requires_rag": False,
            "with_context": False
        }
    
    return {
        "name": CHAT_TYPES.get(chat_type, chat_type),
        "description": PROMPT_MAP[chat_type].get("description", ""),
        "requires_rag": PROMPT_MAP[chat_type].get("requires_rag", True),
        "with_context": PROMPT_MAP[chat_type].get("with_context", True)
    }

def list_available_chat_types() -> list:
    """List all available chat types with descriptions"""
    result = []
    for chat_type, display_name in CHAT_TYPES.items():
        prompt_info = PROMPT_MAP.get(chat_type, {})
        result.append({
            "chat_type": chat_type,
            "display_name": display_name,
            "description": prompt_info.get("description", ""),
            "requires_rag": prompt_info.get("requires_rag", True),
            "with_context": prompt_info.get("with_context", True)
        })
    return result