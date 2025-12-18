"""
Chat Service for Single Analysis Follow-ups
Allows users to ask questions about their single analysis results
"""

from typing import Dict, Any, Optional
from app.services.comparative_analysis import safe_generate
from app.services.single_analysis_rag import generate_with_rag_citations


async def generate_single_analysis_chat_response(
    user_question: str,
    analysis_data: Dict[str, Any],
    domain: Optional[str] = None,
    top_k: int = 5,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """
    Generate a chat response based on single analysis data with RAG support
    
    Args:
        user_question: The user's follow-up question
        analysis_data: The original analysis result containing:
            - response: Original analysis text
            - chat_type: Type of analysis performed
            - citations: Research citations used
            - file_metadata: Files that were analyzed
            - content_analysis: Content statistics
        domain: Optional domain filter for RAG search
        top_k: Number of research chunks to retrieve
        max_tokens: Maximum tokens in response
        
    Returns:
        Dict containing answer, citations, and context used
    """
    
    original_response = analysis_data.get("response", "")
    chat_type = analysis_data.get("chat_type", "ANALYSIS")
    file_metadata = analysis_data.get("file_metadata", [])
    content_analysis = analysis_data.get("content_analysis", {})
    original_citations = analysis_data.get("citations", [])
    
    
    system_prompt = f"""You are an expert analysis assistant helping with follow-up questions about a previously completed {chat_type} analysis.

## Original Analysis Context:

The user previously uploaded {len(file_metadata)} file(s) for analysis:
{_format_file_list(file_metadata)}

The analysis was {content_analysis.get('total_words', 'unknown')} words and contained detailed insights.

## Your Role:

1. Answer the user's follow-up question based on:
   - The original analysis findings (provided below)
   - Additional research evidence from the knowledge base
   - Your expertise in philanthropy and nonprofit evaluation

2. **MANDATORY: Use APA 7 inline citation format**
   - ONLY cite sources with author names using parentheses: (Author, Year)
   - Example: (Latessa et al., 2002)
   - Multiple sources: (Smith, 2020; Jones et al., 2019)
   - **If no author name is available, do NOT include inline citation at all**
   - Do NOT use brackets or numbers like [1] or [2, 2014]

3. Reference specific points from the original analysis when relevant

4. If the question requires information not in the original analysis, search the research corpus for relevant evidence

5. Be concise but thorough - this is a conversation, not a full report

## Original Analysis Summary:
{_extract_key_findings(original_response)}

---

Now answer the user's follow-up question with evidence-based insights."""

    result = await generate_with_rag_citations(
        system_prompt=system_prompt,
        user_query=user_question,
        top_k_research=top_k,
        domain=domain,
        max_tokens=max_tokens,
        enable_web_fallback=False  
    )
    
    result["original_analysis_type"] = chat_type
    result["files_analyzed"] = [f["filename"] for f in file_metadata]
    result["is_follow_up"] = True
    
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
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a simple chat response without RAG (fallback)
    
    Uses only the original analysis data to answer questions
    """
    
    original_response = analysis_data.get("response", "")
    chat_type = analysis_data.get("chat_type", "ANALYSIS")
    file_metadata = analysis_data.get("file_metadata", [])
    
    system_msg = f"""You are an expert analysis assistant. Answer the user's question based on the analysis provided below.

## Context:
- Analysis Type: {chat_type}
- Files Analyzed: {', '.join([f.get('filename', 'Unknown') for f in file_metadata])}

## Original Analysis:
{original_response[:4000]}

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
        "original_analysis_type": chat_type
    }