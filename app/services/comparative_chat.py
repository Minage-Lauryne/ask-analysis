import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def retrieve_relevant_context(
    user_question: str,
    rfp_text: str,
    proposals: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """Retrieve the most relevant context from RFP and proposals"""
    from app.services.comparative_analysis import embed_text, chunk_text
    
    logger.info(f"Retrieving context for question: {user_question[:100]}...")
    question_embedding = embed_text(user_question)
    
    rfp_chunks = chunk_text(rfp_text, chunk_size=800, overlap=100)
    
    rfp_embeddings = np.vstack([embed_text(chunk) for chunk in rfp_chunks])
    
    rfp_scores = (rfp_embeddings @ question_embedding).astype(float)
    top_rfp_indices = np.argsort(rfp_scores)[-top_k:][::-1]
    
    relevant_rfp_chunks = [rfp_chunks[i] for i in top_rfp_indices if i < len(rfp_chunks)]
    logger.debug(f"Selected {len(relevant_rfp_chunks)} most relevant RFP chunks")
    
    relevant_proposals = []
    
    for proposal in proposals:
        analysis_data = proposal.get("analysis", {})
        alignment = analysis_data.get("alignment", {})
        evidence = analysis_data.get("evidence", {})
        
        proposal_context = build_proposal_context(proposal, alignment, evidence)
        
        if proposal_context:
            context_embedding = embed_text(proposal_context)
            similarity_score = float(context_embedding @ question_embedding)
            
            relevant_proposals.append({
                "organization_name": proposal.get("organization_name", "Unknown"),
                "filename": proposal.get("filename", ""),
                "recommendation": proposal.get("recommendation", ""),
                "score": proposal.get("overall_alignment_score", 0),
                "budget": proposal.get("budget", ""),
                "timeline": proposal.get("timeline", ""),
                "risk_level": proposal.get("verification", {}).get("risk_level", "UNKNOWN"),
                "alignment_analysis": alignment,
                "evidence": evidence,
                "relevant_context": proposal_context[:1000], 
                "similarity_score": similarity_score,
                "key_issues": extract_key_issues_from_proposal(proposal)
            })
    
    relevant_proposals.sort(key=lambda x: x["similarity_score"], reverse=True)
    logger.info(f"Found {len(relevant_proposals)} relevant proposals, returning top {min(top_k, len(relevant_proposals))}")
    
    return {
        "rfp_context": "\n\n".join(relevant_rfp_chunks),
        "proposals_context": relevant_proposals[:top_k]
    }


def build_proposal_context(proposal: Dict[str, Any], alignment: Dict[str, Any], evidence: Dict[str, Any]) -> str:
    """Build context string from proposal analysis data"""
    context_parts = []
    
    context_parts.append(f"Organization: {proposal.get('organization_name', 'Unknown')}")
    context_parts.append(f"Recommendation: {proposal.get('recommendation', 'Consider')}")
    context_parts.append(f"Score: {proposal.get('overall_alignment_score', 0)}/100")
    context_parts.append(f"Budget: {proposal.get('budget', 'Not specified')}")
    context_parts.append(f"Timeline: {proposal.get('timeline', 'Not specified')}")
    
    if alignment:
        context_parts.append("\nALIGNMENT ANALYSIS:")
        if alignment.get('what_text'):
            context_parts.append(f"What: {alignment['what_text']}")
        if alignment.get('how_text'):
            context_parts.append(f"How: {alignment['how_text']}")
        if alignment.get('who_text'):
            context_parts.append(f"Who: {alignment['who_text']}")
        if alignment.get('why_text'):
            context_parts.append(f"Why: {alignment['why_text']}")
    
    if evidence:
        context_parts.append("\nEVIDENCE:")
        for category in ['what', 'how', 'who']:
            if evidence.get(category):
                context_parts.append(f"{category.upper()}: {', '.join(evidence[category][:2])}")
    
    verification = proposal.get("verification", {})
    if verification:
        context_parts.append("\nVERIFICATION:")
        context_parts.append(f"Risk Level: {verification.get('risk_level', 'UNKNOWN')}")
        context_parts.append(f"Verified: {verification.get('verified', False)}")
        if verification.get('revenue') is not None:
            context_parts.append(f"Revenue: ${verification.get('revenue', 0):,}")
        if verification.get('assets') is not None:
            context_parts.append(f"Assets: ${verification.get('assets', 0):,}")
    
    return "\n".join(context_parts)


def extract_key_issues_from_proposal(proposal: Dict[str, Any]) -> List[str]:
    """Extract key issues that led to the recommendation"""
    issues = []
    
    budget = proposal.get('budget', '')
    if 'Not specified' in budget:
        issues.append("Budget not specified")
    elif '$' in budget and 'Not specified' not in budget:
        issues.append(f"Budget: {budget}")
    
    risk_level = proposal.get('verification', {}).get('risk_level', '')
    if risk_level in ['HIGH', 'CRITICAL']:
        issues.append(f"High risk level: {risk_level}")
    
    revenue = proposal.get('verification', {}).get('revenue', 0)
    assets = proposal.get('verification', {}).get('assets', 0)
    if revenue == 0 and assets == 0:
        issues.append("No financial capacity ($0 revenue and assets)")
    
    alignment = proposal.get('alignment', {})
    if not alignment.get('what_aligned', True):
        issues.append("Poor alignment with requirements")
    if not alignment.get('who_aligned', True):
        issues.append("Organizational capacity concerns")
    
    score = proposal.get('overall_alignment_score', 0)
    if score < 50:
        issues.append("Low alignment score")
    
    return issues


def generate_chat_response_from_data(
    user_question: str,
    rfp_text: str,
    rfp_title: str,
    rfp_summary: Dict[str, Any],
    proposals: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate AI response based on provided RFP and proposal data"""
    from app.services.comparative_analysis import safe_generate
    
    logger.info(f"Generating chat response for: {user_question[:100]}...")
    
    if not rfp_text:
        logger.warning("No RFP text provided for chat response")
        return {
            "answer": "Sorry, I couldn't find the RFP data.",
            "context_used": None
        }
    
    context = retrieve_relevant_context(user_question, rfp_text, proposals, top_k=3)
    
    question_type = classify_question_type(user_question)
    logger.debug(f"Question classified as: {question_type}")
    
    proposals_summary = ""
    for prop in context["proposals_context"]:
        proposals_summary += f"""
ORGANIZATION: {prop['organization_name']}
- Status: {prop['recommendation']}
- Score: {prop['score']}/100
- Budget: {prop['budget']}
- Timeline: {prop['timeline']}
- Risk Level: {prop['risk_level']}

Key Analysis:
{prop['relevant_context'][:800]}...

---
"""
    
    system_msg = f"""You are an expert RFP analysis assistant. Your role is to help users understand proposal analysis results.

CRITICAL RULES:
1. Answer ONLY based on the provided context - do not make up information
2. Reference specific organizations by name when discussing proposals
3. Be concise but comprehensive in explanations
4. If you don't know something from the context, say so
5. Focus on the analysis data provided - alignment scores, recommendations, budgets, risk levels

RFP TITLE: {rfp_title}
RFP SUMMARY: {rfp_summary.get('executive_summary', 'N/A')}
ANALYSIS CRITERIA: {', '.join(rfp_summary.get('key_requirements', []))}

You have access to detailed analysis of each proposal including:
- Alignment scores and recommendations
- Budget and timeline information
- Risk assessments and organizational capacity
- Detailed alignment analysis (WHAT, HOW, WHO, WHY)
- Evidence from proposals"""

    user_msg = f"""CONTEXT FOR YOUR RESPONSE:

RFP REQUIREMENTS:
{context['rfp_context']}

ANALYZED PROPOSALS:
{proposals_summary}

USER QUESTION: {user_question}

IMPORTANT: 
- Reference specific organizations when answering
- Use the scores, recommendations, and analysis data provided
- Explain reasons clearly based on the context
- If asking about a specific organization, make sure it's in the proposals list above"""

    answer = safe_generate(system_msg, user_msg, max_tokens=1500)
    
    if not answer:
        return {
            "answer": "I apologize, but I'm having trouble generating a response right now. Please try again.",
            "context_used": None
        }
    
    logger.info(f"Successfully generated chat response ({len(answer)} chars)")
    return {
        "answer": answer,
        "context_used": {
            "rfp_chunks_used": len(context["rfp_context"].split("\n\n")),
            "proposals_referenced": [p["organization_name"] for p in context["proposals_context"]],
            "question_type": question_type
        }
    }


def classify_question_type(question: str) -> str:
    """Classify the type of question for better handling"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['why', 'reason', 'not recommended', 'rejected']):
        return "explanation"
    elif any(keyword in question_lower for keyword in ['compare', 'difference', 'better', 'best', 'worst']):
        return "comparison"
    elif any(keyword in question_lower for keyword in ['budget', 'cost', 'price', 'money']):
        return "budget"
    elif any(keyword in question_lower for keyword in ['score', 'rating', 'points']):
        return "scoring"
    elif any(keyword in question_lower for keyword in ['risk', 'safe', 'danger']):
        return "risk"
    else:
        return "general"