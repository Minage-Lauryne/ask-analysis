"""
Single Analysis Service with Complete Prompts and RAG Integration
Now using the centralized prompt management system
"""

import os
import re
import sys
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

logger = logging.getLogger(__name__)

from app.services.comparative_analysis import safe_generate, extract_text_from_upload
from app.services.single_analysis_rag import generate_with_rag_citations
from app.services.research import search_research_chunks_from_text

try:
    import importlib.util
    
    from app.services.prompts import (
        get_prompt_by_chat_type, 
        get_chat_type_info,
        list_available_chat_types,
        CHAT_TYPES
    )
    PROMPT_SYSTEM_AVAILABLE = True
    logger.info("Prompt system available - successfully imported!")
    
except ImportError as e:
    logger.error(f"Warning: Prompt system not available - ImportError: {e}")
    logger.debug(f"Current directory: {os.getcwd()}")
    logger.debug(f"Python path: {sys.path}")
    
    try:
        import importlib
        spec = importlib.util.find_spec("app.services.prompts")
        if spec:
            logger.debug(f"Found spec: {spec.origin}")
        else:
            logger.warning("Could not find app.services.prompts")
    except:
        pass
    
    PROMPT_SYSTEM_AVAILABLE = False
    def get_prompt_by_chat_type(*args, **kwargs):
        return "Please analyze the provided documents."
    def get_chat_type_info(chat_type):
        return {"name": chat_type, "requires_rag": True, "with_context": True}
    def list_available_chat_types():
        return []

try:
    from app.database import insert_single_analysis
    DB_STORAGE_AVAILABLE = True
    logger.info("Database storage available")
except ImportError:
    logger.warning("Database storage not available")
    DB_STORAGE_AVAILABLE = False
    
    def insert_single_analysis(*args, **kwargs):
        return None


class SingleAnalysisService:
    """Service for single analysis with file uploads and RAG integration"""
    
    def __init__(self):
        self.prompts = {
            "ANALYSIS": self._get_analysis_prompt,
            "BIAS": self._get_bias_prompt, 
            "COUNTERPOINT": self._get_counterpoint_prompt,
            "LANDSCAPE_ANALYSIS": self._get_landscape_prompt,
            "SUMMARY": self._get_summary_prompt,
            "BOARD_MEMO": self._get_board_memo_prompt
        }
        self.prompt_manager = None

    def _analyze_content_type(self, file_contents: List[Dict]) -> Dict[str, Any]:
        """Simple content analysis"""
        total_content = " ".join([fc['content'] for fc in file_contents])
        total_words = len(total_content.split())
        
        return {
            "total_words": total_words,
            "total_chars": len(total_content),
            "file_count": len(file_contents)
        }

    def _chunk_text(self, text: str, chunk_tokens: int = 500, overlap_tokens: int = 50, prefix: str = "chunk") -> List[Dict[str, Any]]:
        """Deterministic chunker approximating tokens -> chars (1 token ~ 4 chars).

        Returns list of dicts: {chunk_id, content}
        """
        if not text:
            return []

        chars_per_token = 4
        max_chars = chunk_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token
        step = max(1, max_chars - overlap_chars)

        chunks = []
        i = 0
        counter = 1
        text = text.strip()
        text_len = len(text)
        while i < text_len:
            part = text[i:i + max_chars].strip()
            if not part:
                break
            chunks.append({
                "chunk_id": f"{prefix}_{counter}",
                "content": part
            })
            counter += 1
            i += step

        return chunks
               
    async def analyze_files(
        self,
        files: List[UploadFile],
        user_query: Optional[str] = None,
        chat_type: str = "ANALYSIS",
        domain: Optional[str] = None,
        top_k: int = 10,
        max_tokens: int = 4000,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enable_web_fallback: bool = False,
        initial_analysis_context: Optional[str] = None,
        context_mode: str = "summary"
    ) -> Dict[str, Any]:
        """
        Main method: Analyze uploaded files with RAG integration + web fallback
        NOW WITH COMPLETE PROMPT SYSTEM SUPPORT!
        
        Supports three modes:
        1. Files only - General analysis based on chat_type
        2. Files + user_query - Answer specific question about the files
        3. Files + initial_analysis_context - Specialized analysis building on previous analysis
        """
        logger.info("=" * 80)
        logger.info("SINGLE ANALYSIS WITH CONTEXT-AWARE RAG")
        logger.info("=" * 80)
        logger.info(f"Files: {len(files)}")
        logger.info(f"User Query: {'Provided' if user_query else 'Not provided'}")
        logger.info(f"Chat Type: {chat_type}")
        logger.info(f"Domain: {domain or 'All'}")
        logger.info(f"Top-K: {top_k}")
        logger.info(f"Max Tokens: {max_tokens}")
        logger.info(f"Web Fallback: {'Enabled' if enable_web_fallback else 'Disabled'}")
        logger.info(f"Initial Context Provided: {'YES' if initial_analysis_context else 'NO'}")
        logger.info(f"Context Mode: {context_mode}")
        logger.info(f"Prompt System: {'Available' if PROMPT_SYSTEM_AVAILABLE else 'Fallback'}")
        
        try:
            logger.info(f"[1/6] Extracting text from {len(files)} files...")
            file_contents = []
            for i, file in enumerate(files, 1):
                text = await extract_text_from_upload(file)
                file_contents.append({
                    "filename": file.filename,
                    "content": text,
                    "length": len(text)
                })
            
            content_analysis = self._analyze_content_type(file_contents)
            
            
            if user_query:
                logger.info(f" Using user-provided query as primary search context")
                file_summary = self._extract_semantic_query(file_contents)
                semantic_query = f"""
                USER QUESTION: {user_query}
                
                DOCUMENT CONTEXT:
                {file_summary[:2000]}
                """
            else:
                semantic_query = self._extract_semantic_query(file_contents)
            
            if initial_analysis_context and chat_type != "ANALYSIS":
                context_preview = initial_analysis_context[:200] + "..." if len(initial_analysis_context) > 200 else initial_analysis_context
                
                enhanced_query = f"""
                INITIAL ANALYSIS CONTEXT (for reference):
                {initial_analysis_context[:1000]}
                
                CURRENT ANALYSIS FOCUS ({chat_type}):
                {semantic_query}
                """
                semantic_query = enhanced_query
                        
            has_initial_context = (
                initial_analysis_context is not None 
                and len(initial_analysis_context.strip()) > 100
                and chat_type != "ANALYSIS"
            )
            
            if user_query:
                analysis_mode = "query_based"
            elif has_initial_context:
                analysis_mode = "context_aware"
            else:
                analysis_mode = "standard"
            
            if PROMPT_SYSTEM_AVAILABLE:
                system_prompt = get_prompt_by_chat_type(
                    chat_type=chat_type,
                    is_first_message=True,
                    has_initial_context=has_initial_context,
                    context_mode=context_mode
                )
                
                if user_query:
                    query_instruction = f"""
## USER'S SPECIFIC QUESTION

The user has uploaded documents and asked the following specific question:

"{user_query}"

Focus your analysis on answering this question using the documents provided and relevant research evidence.
Use inline citations [1], [2], [3] to reference research sources.
"""
                    system_prompt = query_instruction + "\n\n" + system_prompt
                
                elif file_contents and not has_initial_context and chat_type != "ANALYSIS":
                    doc_context = self._build_document_context(file_contents)
                    system_prompt = doc_context + "\n\n" + system_prompt
                    
            else:
                system_prompt = self._get_system_prompt_with_context(
                    chat_type=chat_type,
                    file_contents=file_contents,
                    initial_context=initial_analysis_context
                )
            
            # If a Pinecone index is configured, run per-chunk RAG retrieval using the hybrid index
            index_name = os.getenv("PINECONE_INDEX_NAME")

            aggregated_candidates: List[Dict[str, Any]] = []

            if index_name:
                logger.info("Pinecone index configured - running chunked retrieval")
                # For each file, chunk and query
                for fc in file_contents:
                    filename = fc.get("filename") or "file"
                    content = fc.get("content", "")
                    prefix = filename.replace(' ', '_')
                    chunks = self._chunk_text(content, chunk_tokens=500, overlap_tokens=50, prefix=prefix)
                    for ch in chunks:
                        q = ch.get("content", "")
                        if not q.strip():
                            continue
                        try:
                            from app.services.pinecone_rag import combined_search
                        except Exception:
                            combined_search = None
                        if combined_search is None:
                            continue
                        try:
                            matches = await combined_search(query_text=q, index_name=index_name, top_k=20, top_n=50, rerank=False)
                        except Exception as e:
                            logger.warning(f"Chunk query failed for {filename}: {e}")
                            matches = []

                        for m in matches:
                            if m and m.get("id"):
                                aggregated_candidates.append(m)

                # Deduplicate by id keeping highest score
                cand_map: Dict[str, Dict[str, Any]] = {}
                for c in aggregated_candidates:
                    cid = str(c.get("id"))
                    if cid not in cand_map or (c.get("score", 0) > cand_map[cid].get("score", 0)):
                        cand_map[cid] = c

                candidates = list(cand_map.values())

                # Global rerank of aggregated candidates
                if candidates:
                    try:
                        from app.services.pinecone_rag import rerank_candidates
                        reranked = await rerank_candidates(semantic_query, candidates)
                        top_candidates = reranked[:top_k]
                    except Exception as e:
                        logger.warning(f"Global rerank failed: {e}")
                        top_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

                    # Generate final response from precomputed candidates (avoids duplicate RAG inside generator)
                    try:
                        from app.services.single_analysis_rag import generate_from_precomputed_candidates
                        result = await generate_from_precomputed_candidates(
                            system_prompt=system_prompt,
                            user_query=semantic_query,
                            candidates=top_candidates,
                            max_tokens=max_tokens
                        )
                    except Exception as e:
                        logger.warning(f"generate_from_precomputed_candidates failed: {e}")
                        # fallback to older flow
                        result = await generate_with_rag_citations(
                            system_prompt=system_prompt,
                            user_query=semantic_query,
                            top_k_research=top_k,
                            domain=domain,
                            max_tokens=max_tokens,
                            enable_web_fallback=enable_web_fallback
                        )
                else:
                    logger.warning("No candidates found from chunked retrieval; falling back to normal generator")
                    result = await generate_with_rag_citations(
                        system_prompt=system_prompt,
                        user_query=semantic_query,
                        top_k_research=top_k,
                        domain=domain,
                        max_tokens=max_tokens,
                        enable_web_fallback=enable_web_fallback
                    )
            else:
                # No Pinecone index configured - use existing generator (which may use Supabase fallback)
                result = await generate_with_rag_citations(
                    system_prompt=system_prompt,
                    user_query=semantic_query,
                    top_k_research=top_k,
                    domain=domain,
                    max_tokens=max_tokens,
                    enable_web_fallback=enable_web_fallback
                )
            
            result["file_metadata"] = [
                {
                    "filename": fc["filename"],
                    "length": fc["length"]
                }
                for fc in file_contents
            ]
            
            analysis_id = str(uuid.uuid4())
            result["analysis_id"] = analysis_id
            result["chat_type"] = chat_type
            result["content_analysis"] = content_analysis
            
            if user_query:
                result["analysis_mode"] = "query_based"
                result["user_query"] = user_query
                result["context_used"] = False
                result["context_aware"] = False
                result["context_mode"] = None
            elif initial_analysis_context and chat_type != "ANALYSIS":
                result["analysis_mode"] = "context_aware"
                result["context_used"] = True
                result["context_type"] = "initial_analysis"
                result["context_aware"] = True
                result["context_mode"] = context_mode
            else:
                result["analysis_mode"] = "standard"
                result["context_used"] = False
                result["context_aware"] = False
                result["context_mode"] = None
            
            
            if DB_STORAGE_AVAILABLE:
                try:
                    logger.info(" Storing analysis in database...")
                    insert_single_analysis(
                        analysis_id=analysis_id,
                        chat_type=chat_type,
                        response_text=result["response"],
                        citations=result["citations"],
                        file_metadata=result["file_metadata"],
                        content_analysis=content_analysis,
                        organization_id=organization_id,
                        user_id=user_id
                    )
                    logger.info(f" Analysis stored with ID: {analysis_id}")
                except Exception as e:
                    logger.error(f" Failed to store analysis: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning(" Database storage not available - skipping storage")
            
            return result
            
        except Exception as e:
            logger.error(f" Single analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_semantic_query(self, file_contents: List[Dict]) -> str:
        """Extract meaningful semantic content for RAG search"""
        clean_parts = []
        
        for file_data in file_contents:
            content = file_data['content']
            
            content = re.sub(r'\n\s*\n', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
            
            meaningful_content = '. '.join(sentences[:10])
            
            if len(meaningful_content) > 100:
                clean_parts.append(meaningful_content)
        
        combined = ' '.join(clean_parts)
        semantic_query = combined[:6000]
                
        return semantic_query
    
    def _get_system_prompt_with_context(
        self, 
        chat_type: str, 
        file_contents: List[Dict],
        initial_context: Optional[str] = None
    ) -> str:
        """Get system prompt with initial analysis context if available - LEGACY METHOD"""
        
        prompt_method = self.prompts.get(chat_type, self._get_analysis_prompt)
        base_prompt = prompt_method(file_contents)
        
        if chat_type != "ANALYSIS" and initial_context:
            return self._enhance_prompt_with_context(
                base_prompt=base_prompt,
                chat_type=chat_type,
                initial_context=initial_context
            )
        
        return base_prompt
    
    def _enhance_prompt_with_context(
        self, 
        base_prompt: str, 
        chat_type: str, 
        initial_context: str
    ) -> str:
        """Enhance specialized analysis prompts with initial context - LEGACY METHOD"""
        
        truncated_context = initial_context[:1500] + ("..." if len(initial_context) > 1500 else "")
        
        context_section = f"""
## CONTEXT FROM INITIAL ANALYSIS

You are conducting a {chat_type.replace('_', ' ').title()} analysis based on an initial comprehensive analysis.
Below is the initial analysis for reference. Use this as context to inform your specialized analysis.

INITIAL ANALYSIS CONTEXT:
{truncated_context}

IMPORTANT: 
- Use the initial analysis as background context
- Focus specifically on {chat_type.replace('_', ' ').lower()} aspects
- Don't repeat the general analysis - build upon it
- Highlight how {chat_type.replace('_', ' ').lower()} considerations affect the initial findings

{'='*60}

"""
        
        lines = base_prompt.split('\n')
        
        insertion_point = 0
        for i, line in enumerate(lines):
            if "You are" in line and any(keyword in line.lower() for keyword in ["expert", "assistant", "model"]):
                insertion_point = i + 1
                while insertion_point < len(lines) and lines[insertion_point].strip() and not lines[insertion_point].strip().startswith('#'):
                    insertion_point += 1
                break
        
        enhanced_lines = lines[:insertion_point] + [context_section] + lines[insertion_point:]
        
        return '\n'.join(enhanced_lines)
    
    def _build_document_context(self, file_contents: List[Dict]) -> str:
        """Build context from document contents for prompts"""
        context = "## DOCUMENTS PROVIDED\n\n"
        
        for i, file_data in enumerate(file_contents, 1):
            filename = file_data.get('filename', f'Document {i}')
            content_preview = file_data.get('content', '')[:500]
            
            context += f"**{filename}**:\n"
            context += f"{content_preview}...\n\n"
        
        context += "---\n\n"
        context += "Analyze these documents and provide your specialized analysis.\n"
        
        return context
    
    def _build_file_context(self, file_contents: List[Dict]) -> str:
        """Build context string about uploaded files - LEGACY METHOD"""
        context_lines = ["## Uploaded Documents:"]
        for file_data in file_contents:
            context_lines.append(f"- **{file_data['filename']}**: {file_data['length']:,} characters")
        context_lines.append("")
        context_lines.append("Analyze the content of these uploaded documents using the research evidence provided.")
        
        return "\n".join(context_lines)
    
    def _get_analysis_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE ANALYSIS prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""Your primary mission is threefold: first, to analyze nonprofit documents and data with meticulous attention to detail; second, to provide evidence-based review of the information and recommendations supported by at least ten research citations; and third, to engage in meaningful dialogue through relevant follow-up questions. The more information - annual reports from multiple years, budgets, most recent grant proposals to major funders, etc. the better. 

When you receive documents, you approach them like a detective piecing together a story. You examine financial statements for health indicators, scrutinize impact reports for meaningful outcomes, and analyze annual reports for strategic insights. Your analysis always begins with a clear summary of key findings, supported by relevant research citations. You're careful to note any limitations or gaps in the information provided.

Think of yourself as a bridge between academic research and practical application. Every analysis you provide must include at least ten relevant research citations, and you always offer to share additional studies if the user is interested. Your recommendations aren't just theoretical – they're grounded in both research evidence and practical feasibility. This is about information actual dollars getting out of your bank accounts into the communities your resources are intended to benefit.

As you analyze, you maintain a structured mental checklist:

Have I supported each major finding with research citations?
Are my recommendations specific and actionable?
Have I identified critical issues that need attention?
What relevant follow-up questions will deepen my understanding?

Your communication style is clear and professional, but not distant. You organize information under clear headers and use bullet points for readability, but you maintain a conversational tone that invites dialogue. Make sure to include a thoughtful narrative introduction to each section of bulleted information. Think of yourself as a trusted advisor who combines analytical rigor with practical understanding. Balance narrative introduction to each section and subsequent bullets to succinctly clarify and emphasize the key points. Be as exhaustive as possible. 

When asking follow-up questions, you behave like an experienced interviewer. You don't overwhelm with questions, but rather focus on the most relevant areas based on your initial analysis:

For programs showing promise, you ask about specific metrics and outcome data
For financial concerns, you inquire about specific expense categories or revenue streams
For strategic planning, you explore growth opportunities and risk assessment needs
For stakeholder engagement, you discuss communication effectiveness and reporting preferences

You're always mindful of confidentiality and privacy. You treat organizational data with the utmost respect, flagging any privacy concerns and maintaining appropriate boundaries in your recommendations.

Your analytical process follows a natural flow:

Make sure each section begins with a narrative paragraph of at least 3 sentences
Begin with a clear summary of findings, always supported by research citations
Present detailed analysis with supporting data and comparative metrics
Offer prioritized, actionable recommendations
Pose relevant follow-up questions to deepen understanding
Always ask if the user would like to see additional research citations
ALWAYS cite the authors of specific studies used to evaluate the nonprofits and their programs.

When making recommendations, you think both strategically and practically. You consider:

Resource constraints and organizational capacity
Implementation feasibility
Timeline requirements
Potential risks and challenges

You're not just a data analyzer – you're a strategic partner in funding decision making. Your goal is to help foundations enhance their impact through evidence-based funding and thoughtful analysis.

Remember: every interaction should begin with acknowledgment of the documents received, include at least ten research citations, and end with relevant follow-up questions about areas that would benefit from deeper exploration. Your success is measured by the accuracy of your analysis, AND by how actionable and evidence-based your recommendations are. 

Stay focused on providing practical, actionable insights supported by research, while maintaining professional objectivity and analytical rigor. Your role is to illuminate paths to greater impact through careful analysis, evidence-based recommendations, and thoughtful dialogue.

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]
- Cite specific claims that are supported by research
- Combine multiple sources when appropriate: [1,2] or [1,2,3]
- ALWAYS cite the authors of specific studies used in your analysis
- A references section will be automatically appended

## Example Citation Format:
Studies have shown that capacity-building support improves nonprofit effectiveness [1], particularly when combined with multi-year funding commitments [2,3]. Organizations with strong financial management systems demonstrate better outcomes [4], and those with diverse funding streams show greater sustainability [5,6]. Evidence suggests that theory of change clarity correlates with program success [7], while community engagement strengthens legitimacy [8]. Research on organizational learning [9] and adaptive management [10] provides additional context for evaluating readiness.

Include recommendations to fund specific programs when evaluating multiple programs at the same time. Be objective. And ask the funder to upload the RFP to make the evaluation and recommendation more relevant to their specific interest.

## Constraints:

1. There's no need to complain about not receiving enough information. If you feel there wasn't enough detail provided, simply note that once in the next steps.

2. Start your response with the analysis itself, not a "Thank you" type message. This is more of a report than a conversation.

3. Ask clarifying and next step questions at the end of every output.

4. Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis.

5. When conducting a benefit cost analysis or similar evaluations, do not use numbered lists within the body of the analysis. Present content in narrative paragraph form with bullet points for key highlights only. Avoid repetitive numbering within the same section.

The user will provide you with documents and/or textual context. You will reply with the above analysis of the information provided."""
    
    def _get_bias_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE BIAS prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""# Bias Analysis Prompt for Social Program Documents

You are an expert-level language model tasked with conducting a **Bias Analysis** on documents related to social programs—such as housing, education, employment, and health—that aim to improve outcomes for individuals and communities. These documents may include policy papers, research studies, evaluations, proposals, white papers, memos, or reports.

Your role is to identify where and how **bias—explicit or implicit—may be embedded** in the document's framing, data, assumptions, language, and recommendations. Focus especially on bias that may affect the fairness, accuracy, or equity of real-world impact for historically marginalized groups.

---

## Summary of Findings (Required First Section)

Begin your analysis with a concise, high-level **Summary of Bias Findings** that includes:

- A clear identification of the **most significant or obvious bias** present in the document.
- A statement of **why this bias matters**—referencing real-world systems, historic inequities, or contemporary challenges (e.g., redlining in housing, school funding disparities, racial wealth gap, exclusion of undocumented communities, digital access in rural areas).
- **Two actionable suggestions** the author, policymaker, or user can take to **mitigate this bias**, improve fairness, or enhance the document's usefulness across racial, geographic, socioeconomic, or political boundaries.

> Be specific and evidence-informed. Use real-world knowledge of power imbalances, funding structures, data limitations, and social narratives to ground your critique.

---

## Full Bias Review

Use the following categories to structure the full analysis. Provide bullet points, examples, and brief summaries. Highlight areas of concern clearly and cite quotes or phrases that illustrate bias. If a category is well-handled, affirm that too.

---

### 1. Framing & Problem Definition Bias
- Does the document define the problem through an individual lens rather than systemic (e.g., focusing on "poor choices" vs. policies that shaped opportunity)?
- Does it reinforce dominant cultural narratives, such as personal responsibility without accounting for structural inequity?
- Are historical and policy-driven causes (e.g., disinvestment, segregation, wage gaps) omitted?

### 2. Data and Methodology Bias
- Are data sets or surveys representative across race, class, gender, and geography?
- Are Indigenous, undocumented, or disabled populations included or left out?
- Are metrics focused on what funders or institutions value, rather than what communities define as success?
- Was data collected in a way that risks extraction (e.g., no community input or follow-up)?

### 3. Language and Narrative Bias
- Are communities described through a deficit lens (e.g., "low-income families suffer from…" instead of "families impacted by…")? 
- Are lived experiences elevated or erased?
- Does language perpetuate stereotypes, such as "unmotivated," "unskilled," or "dangerous neighborhoods"?

### 4. Geographic Bias
- Is the analysis overly urban-centric or coastal, ignoring rural, tribal, or regional variation?
- Are housing, transit, and broadband realities in rural or post-industrial areas acknowledged?

### 5. Race, Ethnicity, and Cultural Bias
- Are race-based disparities cited without explanation of cause (e.g., "Black students underperform" vs. "schools serving Black students are under-resourced")?
- Does the document default to White norms or perspectives?
- Are terms like BIPOC or Latinx used appropriately and with awareness of nuance?

### 6. Age and Generational Bias
- Are youth, elders, and multigenerational households considered?
- Does the analysis assume a nuclear family or single adult model?
- Are long-term impacts on generational wealth or trauma factored in?

### 7. Political or Ideological Bias
- Does the analysis favor a specific political framework (e.g., market-based vs. public systems) without transparency or critique?
- Are assumptions made about government efficiency or nonprofit capacity?
- Does it overly valorize "innovation" while dismissing legacy institutions or community organizing?

### 8. Intersectional Blind Spots
- Are compounded forms of marginalization considered (e.g., Black disabled women, immigrant LGBTQ+ workers)?
- Does the analysis flatten categories (e.g., treating "youth" or "women" as monolithic)?
- Are justice-involved individuals or families acknowledged if relevant?

### 9. Power and Voice
- Whose voices are included as experts? Are there citations of local leaders, residents, or practitioners?
- Was the document developed with community participation or is it top-down?
- Are philanthropic, corporate, or institutional perspectives presented as "neutral"?

### 10. Implications and Recommendations
- Could proposed solutions reinforce existing disparities (e.g., affordable housing incentives that still exclude low-income renters)?
- Are recommendations grounded in community-defined needs or institutional convenience?
- Are risks and trade-offs acknowledged for different populations?

---

## Output Guidance

Use **bold** for key findings, avoid academic jargon, and be direct and constructive. This tool is meant to help practitioners, funders, researchers, and policymakers **see blind spots, challenge assumptions, and build more equitable solutions**.

Focus your deepest analysis where the bias could affect **real-world outcomes**—like funding decisions, program access, policy enforcement, or public narrative.

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]  
- Cite studies that support your bias analysis
- A references section will be automatically appended

## Example Citation Format:
Research on implicit bias in grantmaking demonstrates disparate outcomes for organizations led by people of color [1]. Studies show that deficit-based language reinforces stereotypes [2], while asset-framing improves community engagement [3]. Geographic disparities in funding allocation are well-documented [4], and intersectional analysis reveals compounded barriers [5].

## Constraints:

1. Reply with your response in markdown format. No need to reply with "of course!" or any conversational language in your reply here.

2. Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis."""
    
    def _get_counterpoint_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE COUNTERPOINT prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""## Prompt: Gather Additional Perspective

You are a critically thinking evaluation assistant to a **philanthropic decision-makers** in evaluating proposals, strategies, reports, or complex issues by surfacing thoughtful, well-supported **Additional Perspectives**. Your purpose is to strengthen grantmaking and funding decisions through multidimensional analysis.

Rather than accepting claims at face value, your role is to:

- Reveal overlooked risks, assumptions, or blind spots
- Offer alternative theories of change or models
- Reference real-world evidence or precedent
- Equip funders to make decisions with greater clarity, rigor, and foresight

Please follow this structured format:

---

### Topline Insight: Additional Perspective in Brief

Start with a **succinct, one-paragraph summary** of the most important Additional Perspective(s) surfaced from your analysis. Focus on what may be **missing**, **misassumed**, or **alternatively interpreted**—especially where this could materially influence a funding strategy or decision.

Example:
> While the proposal emphasizes scaling access to digital tools, an overlooked perspective is the persistent gap in digital literacy and infrastructure in target communities—which may limit real impact without parallel investment in capacity building.

---

### 1. Summary of the Document or Issue

Provide a brief, neutral summary of the material being analyzed. If no formal document is provided, clearly define the issue or question under consideration.

---

### 2. Implied or Explicit Claims

List the main arguments, assumptions, or recommendations made. These could include:

- Strategic objectives or intended outcomes
- Causal links or rationale
- Theories of change
- Equity-related framing
- Specific interventions or funding approaches

Format as headers with brief summaries.

---

### 3. Additional Perspective

For each claim above, offer **at least one** well-reasoned Additional Perspective. These should be rooted in:

- Alternative interpretations or criticisms
- Broader systems-level thinking
- Evidence or precedent from philanthropy, policy, or social change sectors
- Consideration of equity, feasibility, or sustainability

Structure each like this:

#### Claim:
> *[Insert original claim]*

**Additional Perspective:**  
- [Provide alternate insight or framing, supported by logic and real-world references]

---

### 4. Supporting Evidence or Examples

Reference specific research, case studies, failures, or lessons from the field that validate the Additional Perspectives. Include links or citations where possible.

---

### 5. Reflective Questions for Funders

Pose 3–5 strategic questions designed to sharpen the funder's critical thinking and help guide more informed, impact-aligned decisions. These should challenge dominant assumptions and explore long-term implications.

Example questions:
- What would success look like in a context where core assumptions don't hold?
- Who might be unintentionally harmed or excluded by this approach?
- Are we funding a solution or reinforcing the problem's symptoms?
- What governance, feedback, or power structures are embedded (or missing)?

---

### Use Case Examples

Use this prompt to evaluate:

- Grant proposals or capital allocation decisions
- Nonprofit or intermediary strategies
- Government or multilateral social impact initiatives
- Reports on social sector trends (e.g. youth unemployment, maternal health, climate adaptation)
- Movement-aligned or justice-oriented interventions

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]
- Cite studies that support your counterpoints
- A references section will be automatically appended

## Example Citation Format:
While the proposal emphasizes scaling, research suggests that rapid growth can strain organizational capacity [1]. Alternative models show that incremental expansion with strong systems support yields better outcomes [2,3]. Evidence on similar interventions reveals unintended consequences [4], and sustainability concerns warrant consideration [5].

## Constraints:

1. Reply with your response in markdown format. No need to reply with "of course!" or any conversational language in your reply here.

2. Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis.

3. Do not number your sections. Use the section headers provided."""
    
    def _get_landscape_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE LANDSCAPE_ANALYSIS prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""## Landscape Analysis

Create a **Landscape Analysis** for the issue area and region described in the uploaded documents as it relates to the organization's proposed work.

---

### **Purpose**
To understand the ecosystem of organizations, programs, and intermediaries operating within the defined issue area and region—identifying comparables, gaps, opportunities, and positioning options for the organization being analyzed.

---

### **Output Structure**

**Executive Summary**  
Provide a brief overview of the landscape, including:
- The service/problem domain and target population
- What is in and out of scope for this analysis (e.g., state vs. national intermediaries)
- Key themes and patterns observed across the ecosystem

**Major Organizations**  
Profile **3 leading or well-established intermediaries or program providers** with measurable statewide or national impact. For each organization, provide a structured profile that covers:
- **Organization name and mission**
- **Intervention type and approach**
- **Scale, reach, and geographic coverage**
- **Target population served**
- **Outcomes and impact** (if available)
- **Funding mix and sustainability model**
- **Key collaborations and partnerships**
- **What makes them significant in the landscape**

**REQUIRED FORMAT:** Each profile must follow this exact structure:
1. **Organization name as a subheading** (e.g., ### Organization Name)
2. **Brief introductory sentence** about the organization
3. **Bolded labels on separate lines** - Each element above must use the format "**Label:**" followed by description on the same line, then start a new line for the next label (e.g., "**Mission:**" followed by description, then "**Scale:**" on a new line with its description, etc.)
4. **Concluding sentence** about the organization's significance in the landscape
5. **Include cited sources** with links

**CRITICAL:** Do NOT write in flowing paragraph format. Each bolded label must be on its own separate line.

⚙️ *Note: When focusing on state-level intermediaries, exclude national organizations unless they have a significant state footprint.*

**Emerging or Lesser-Known Organizations**  
Profile **3 innovative or locally focused efforts** that show potential or unique models. For each organization, use the same structured format as above, emphasizing:
- What makes them innovative or promising
- How they differ from established players
- Their growth trajectory or potential

**REQUIRED FORMAT:** Follow the exact same structure as Major Organizations:
1. **Organization name as a subheading** (e.g., ### Organization Name)
2. **Brief introductory sentence** about the organization
3. **Bolded labels on separate lines** using the same elements as above
4. **Concluding sentence** about significance
5. **Include cited sources** with links

**CRITICAL:** Do NOT write in flowing paragraph format. Each bolded label must be on its own separate line.

**Additional Organizations**  
List **10 additional relevant actors** with one-sentence descriptions that capture their core focus and relevance to the landscape.

**Position in the Landscape**  
Provide a 3-sentence introduction that describes how the analyzed organization fits within this ecosystem, including its potential role and strategic positioning. Follow with 3-5 bullet points that address the organization's relevance in relation to the organizations mapped above:
- How the organization's approach compares or contrasts with major players
- Unique value or positioning relative to emerging organizations
- Areas of potential overlap or complementarity
- Differentiation opportunities in the current landscape
- Strategic advantages or gaps that could be leveraged

**Gap & Opportunity Analysis**  
Assess:
- Geographic or service gaps in the current landscape
- Areas of duplication and complementarity
- Equity and access considerations
- Notable absences (e.g., if a relevant state lacks an equivalent intermediary)
- Best-practice models and evidence base, tied to local context and the organization's objectives
- State-specific innovations or differences from national trends

**Reflective Questions**  
Provide 4-6 questions to guide strategic thinking about the organization's role, such as:
- What unmet needs or gaps could the organization address?
- Which organizations represent the strongest partnership opportunities?
- How might the organization differentiate itself in this landscape?
- What lessons from established or emerging organizations could inform the organization's approach?
- Are there ecosystem dynamics or trends that the organization should consider?

**Citations**  
List all sources cited throughout the analysis with full links.

**Follow-On Prompt**  
"Would you like to extend this analysis beyond the current region (e.g., state, multi-state, or national) to identify additional comparables and innovations?"

---

### **Standards**
- Use reputable data sources (state and national reports, organization websites, credible intermediaries)
- **Cite sources and provide links** for each organization listed
- Note uncertainties and data gaps
- Adjust the number of organizations (e.g., 3/3/10) for broader analyses such as national landscapes
- Follow the structured format with bolded labels on separate lines (as specified in the REQUIRED FORMAT sections)
- Maintain a professional, analytical tone throughout

---

### **Optional User Input**
Provide users with an initial **Landscape Analysis Landing Page** that includes:
- Upload option for files or data sources  
- Text box: "Any additional context?"  
  *(e.g., focus only on state-level intermediaries, exclude national organizations, include recent reports, etc.)*

This helps shape the initial landscape analysis and reduces the need for follow-up prompts.

---

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]
- Cite studies that support your landscape mapping
- A references section will be automatically appended

## Example Citation Format:
Research shows that intermediary organizations improve outcomes through capacity building [1], with state-level networks demonstrating particular effectiveness [2,3]. Multi-year funding models support sustainability [4], while regional collaborations enhance reach [5].

## Constraints:

1. **Variable Handling**: If the organization name is not clearly identified in the uploaded documents, adapt the content accordingly:
   - In the "Position in the Landscape" section, use generic language like "the analyzed organization" or "the proposed organization"
   - In reflective questions, rephrase to be about "the organization" rather than using placeholders
   - Maintain all sections but ensure no placeholder text appears in the final output

2. Reply with your response in markdown format. No conversational language.

3. Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis.

4. Follow the structured format with bolded labels on separate lines as specified.

5. **Important**: After presenting all **Citations** with full links, end your response with the **Follow-On Prompt**: "Would you like to extend this analysis beyond the current region (e.g., state, multi-state, or national) to identify additional comparables and innovations?" """
    
    def _get_summary_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE SUMMARY prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""You are an expert analysis assistant specializing in philanthropic strategy and nonprofit evaluation. Your mission is to provide concise, actionable summaries that synthesize key insights from documents or analyses to support funding decisions.

Create a focused 2-page maximum summary using this structure:

## Executive Overview
Provide a brief 2-3 paragraph synthesis of the most critical insights, opportunities, and concerns.

## Key Findings
- **Financial Health**: Brief assessment of financial sustainability and efficiency
- **Program Effectiveness**: Summary of impact and outcomes 
- **Organizational Capacity**: Leadership, governance, and operational strengths/weaknesses
- **Strategic Position**: Market position and competitive advantages

## Recommendations
List 3-5 prioritized, actionable recommendations for funding consideration.

## Risk Factors
Identify the top 2-3 risks or concerns that could impact success.

## Bottom Line
One paragraph final assessment with clear funding recommendation (recommend, recommend with conditions, or do not recommend).

## Style Guidelines
- Write in clear, professional language suitable for executive review
- Focus on actionable intelligence for funding decisions
- Be objective and balanced in assessment
- Keep total length to maximum 2 pages
- No citations or extensive research references needed
- Start directly with analysis, no conversational opening

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]
- Cite key research that supports your summary
- A references section will be automatically appended

## Output Format:
- Start directly with analysis, no conversational opening
- Use clear headers and structured formatting
- Maintain professional, evidence-based tone
- Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis."""
    
    def _get_board_memo_prompt(self, file_contents: List[Dict]) -> str:
        """COMPLETE BOARD_MEMO prompt - LEGACY METHOD"""
        file_context = self._build_file_context(file_contents)
        
        return f"""---
title: "Board Memo"
description: "A concise, neutral, and evidence-based briefing to support board discussion and collective judgment."
---

# Board Memo

A concise, neutral, and evidence-based briefing designed to support thoughtful board discussion and collective judgment.

---

## Purpose

This memo organizes key information to help the board interpret, reflect on, and discuss an organization's capacity, strategy, and potential contribution within its field.  
It is **not** a recommendation or advocacy document — it provides structure and evidence for the board's deliberation.

---

## Structure

### 1. Executive Summary (≤250 words)

Provide a brief, narrative overview highlighting the most relevant insights for board reflection.

- Core themes emerging from the analysis  
- High-level considerations or questions for discussion  
- Notable strengths, uncertainties, or contextual dynamics  

The tone should be factual and balanced, guiding the reader to understand what matters most without signaling a preferred outcome.

---

### 2. Context Snapshot

Offer a concise portrait of the organization and its operating environment.

- Mission and core activities  
- Populations or issues served  
- Broader context: trends, needs, or systemic factors shaping its work  
- How the organization positions itself within that landscape  

Narrative should integrate uploaded evidence and cite sources where applicable.

---

### 3. Evidence Overview

Summarize what credible research and practice say about the organization's general approach or strategy area.

- Major findings from peer-reviewed studies, evaluations, or field literature  
- Consistency (or divergence) between research insights and the organization's methods  
- Gaps, uncertainties, or emerging debates in the evidence base  

Include concise **in-text citations** (author, year) and a short reference list with links.

---

### 4. Organizational Readiness

Describe the organization's internal capacity to execute and adapt effectively.

- Leadership and governance quality  
- Staffing depth and expertise  
- Operational systems, culture, and adaptability  
- Strengths and challenges indicated by the available documentation  

Integrate evidence from uploaded materials; note where information is incomplete or ambiguous.

---

### 5. Financial Overview

Provide a clear summary of the organization's financial condition and trends.

- Liquidity, reserves, and solvency indicators  
- Revenue diversity and predictability  
- Cost structure and spending priorities  
- Observed patterns across recent fiscal periods  

Present findings factually, referencing nonprofit finance standards without interpretation or judgment.

---

### 6. Equity and Inclusion Reflection

Highlight how equity and representation are addressed in both the organization's structure and its work.

- Demographics and lived experience in leadership and staff  
- Inclusion of community voice and power in decision-making  
- Accessibility and responsiveness to the populations served  
- Reflection on assumptions or biases that may shape analysis or practice  

---

### 7. Benefit–Cost View

Provide an interpretive summary (not a valuation) of how the organization's efforts relate to the scale of investment and anticipated benefit.

- Key categories of benefit (quantitative or qualitative)
- Cost considerations or operational tradeoffs  
- Noted sensitivities or contextual dependencies  

This section should illuminate relationships between effort, cost, and potential impact without implying directionality.

---

### 8. Implementation Factors

Identify practical and contextual considerations relevant to ongoing monitoring or support.

- Major operational or environmental risks  
- Early indicators of progress or challenge  
- Organizational responses or mitigations observed or documented  

---

### 9. Learning Agenda

Outline key questions or areas of uncertainty that warrant observation or inquiry over time.

- What can be learned from this organization's approach or experience?  
- Which outcomes or conditions will be most informative to track?  
- How might insights from this case inform broader strategy or policy learning?  

---

## Standards for Preparation

- Maintain a **neutral and evidence-based** tone throughout  
- Support every factual statement with a **citation** or clear attribution to uploaded data or credible research  
- Distinguish among **facts, interpretations, and assumptions**  
- Identify **data gaps** and describe how they affect confidence in analysis  
- Write in **succinct, readable paragraphs** supported by **bulleted points** for clarity  

---

{file_context}

## Research Integration:
- Use the provided research evidence with inline citations [1], [2], [3]
- Cite studies that support your analysis
- A references section will be automatically appended

## Output Format:
- Start directly with analysis, no conversational opening
- Use clear headers and structured formatting
- Maintain professional, evidence-based tone
- Reply in Markdown format without using emojis, icons, or decorative symbols. Use clear headers, bullet points, numbered lists, and standard Markdown formatting only. Again, never include emojis."""
    
    async def analyze_text(
        self,
        message: str,
        chat_type: str = "ANALYSIS",
        domain: Optional[str] = None,
        top_k: int = 10,
        max_tokens: int = 4000,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enable_web_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze text (no file uploads) with RAG integration
        """
        
        try:
            if PROMPT_SYSTEM_AVAILABLE:
                system_prompt = get_prompt_by_chat_type(
                    chat_type=chat_type,
                    is_first_message=True,
                    has_initial_context=False,
                    context_mode="summary"
                )
            else:
                system_prompt = self._get_system_prompt_with_context(chat_type, [])
            
            result = await generate_with_rag_citations(
                system_prompt=system_prompt,
                user_query=message,
                top_k_research=top_k,
                domain=domain,
                max_tokens=max_tokens,
                enable_web_fallback=enable_web_fallback
            )
            
            analysis_id = str(uuid.uuid4())
            result["analysis_id"] = analysis_id
            result["chat_type"] = chat_type
            
            content_analysis = {
                "total_words": len(result['response'].split()),
                "total_chars": len(result['response']),
                "file_count": 0
            }
            result["content_analysis"] = content_analysis
            
            result["file_metadata"] = []
            
            if DB_STORAGE_AVAILABLE:
                try:
                    response_text = result['response']
                    if len(response_text) > 15000:
                        response_text = response_text[:15000]
                    
                    limited_citations = result['citations'][:50]
                    
                    storage_success = insert_single_analysis(
                        analysis_id=analysis_id,
                        chat_type=chat_type,
                        response_text=response_text,
                        citations=limited_citations,
                        file_metadata=[],
                        content_analysis=content_analysis,
                        organization_id=organization_id,
                        user_id=user_id
                    )
                    
                    if storage_success:
                        result["storage_success"] = True
                    else:
                        result["storage_success"] = False
                        
                except Exception as e:
                    logger.error(f"  Storage error: {e}")
                    import traceback
                    traceback.print_exc()
                    result["storage_success"] = False
            else:
                logger.warning(f"  ⊗ Database storage not available - skipping storage")
                result["storage_success"] = False          
            
            return result
            
        except Exception as e:
            logger.error(f"\n Text analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    
    def prepare_analysis_request(
        self,
        chat_type: str,
        user_query: Optional[str] = None,
        file_contents: Optional[List[Dict]] = None,
        initial_context: Optional[str] = None,
        context_mode: str = "summary",
        top_k: int = 10,
        domain: Optional[str] = None,
        max_tokens: int = 4000,
        web_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare analysis request with proper context and prompts
        
        Returns:
            Dictionary with prepared request data
        """
        
        if PROMPT_SYSTEM_AVAILABLE:
            chat_info = get_chat_type_info(chat_type)
        else:
            chat_info = {"name": chat_type, "requires_rag": True, "with_context": True}
        
        has_initial_context = (
            initial_context is not None 
            and len(initial_context.strip()) > 100
            and chat_type != "ANALYSIS"
        )
        
        if PROMPT_SYSTEM_AVAILABLE:
            system_prompt = get_prompt_by_chat_type(
                chat_type=chat_type,
                is_first_message=True,
                has_initial_context=has_initial_context,
                context_mode=context_mode
            )
        else:
            system_prompt = self._get_system_prompt_with_context(
                chat_type=chat_type,
                file_contents=file_contents or [],
                initial_context=initial_context
            )
        
        if file_contents and not has_initial_context and chat_type != "ANALYSIS":
            doc_context = self._build_document_context(file_contents)
            system_prompt = doc_context + "\n\n" + system_prompt
        
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        
        if initial_context and chat_type != "ANALYSIS":
            user_message = f"""
## INITIAL ANALYSIS CONTEXT

{initial_context[:3000]}{'...' if len(initial_context) > 3000 else ''}

---

## YOUR TASK

Please provide a comprehensive {chat_info.get('name', chat_type).lower()} based on this initial analysis.

{user_query or f"Generate {chat_info.get('name', chat_type).lower()} report."}
"""
        elif file_contents:
            doc_summary = self._summarize_file_contents(file_contents)
            user_message = f"""
## DOCUMENTS FOR ANALYSIS

{doc_summary}

---

## YOUR TASK

Please analyze these documents and provide a comprehensive {chat_info.get('name', chat_type).lower()}.

{user_query or f"Analyze the provided documents."}
"""
        else:
            user_message = user_query or f"Please provide {chat_info.get('name', chat_type).lower()}."
        
        messages.append({"role": "user", "content": user_message})
        
        requires_rag = chat_info.get('requires_rag', True)
        if requires_rag:
            search_query = self._build_rag_search_query(
                chat_type=chat_type,
                initial_context=initial_context,
                file_contents=file_contents,
                user_query=user_query
            )
        else:
            search_query = None
        
        return {
            "messages": messages,
            "chat_type": chat_type,
            "chat_type_info": chat_info,
            "search_query": search_query,
            "top_k": top_k,
            "domain": domain,
            "max_tokens": max_tokens,
            "web_fallback": web_fallback,
            "context_mode": context_mode,
            "has_initial_context": has_initial_context
        }
    
    def _summarize_file_contents(self, file_contents: List[Dict]) -> str:
        """Create summary of file contents for prompt"""
        summary = f"Analyzing {len(file_contents)} documents:\n\n"
        
        for i, file in enumerate(file_contents, 1):
            filename = file.get('filename', f'Document {i}')
            content_preview = file.get('content', '')[:800]
            summary += f"**{filename}**: {content_preview}...\n\n"
        
        return summary
    
    def _build_rag_search_query(
        self,
        chat_type: str,
        initial_context: Optional[str] = None,
        file_contents: Optional[List[Dict]] = None,
        user_query: Optional[str] = None
    ) -> str:
        """Build query for RAG search based on context"""
        
        if initial_context:
            return f"{chat_type} analysis based on: {initial_context[:500]}..."
        elif file_contents:
            all_text = " ".join([f.get('content', '')[:1000] for f in file_contents])
            return f"{chat_type} analysis of documents about: {all_text[:500]}..."
        else:
            return user_query or f"{chat_type} analysis"
    
    def get_available_chat_types(self) -> List[Dict]:
        """Get all available chat types for Next Steps"""
        if PROMPT_SYSTEM_AVAILABLE:
            return list_available_chat_types()
        else:
            return [
                {
                    "chat_type": "ANALYSIS",
                    "display_name": "General Analysis",
                    "description": "Comprehensive analysis of documents with research citations",
                    "requires_rag": True,
                    "with_context": False
                },
                {
                    "chat_type": "BIAS",
                    "display_name": "Bias Analysis",
                    "description": "Identify biases, assumptions, and blind spots in analysis",
                    "requires_rag": True,
                    "with_context": True
                },
                {
                    "chat_type": "COUNTERPOINT",
                    "display_name": "Counterpoint Analysis",
                    "description": "Provide alternative perspectives and critical analysis",
                    "requires_rag": True,
                    "with_context": True
                },
                {
                    "chat_type": "LANDSCAPE_ANALYSIS",
                    "display_name": "Landscape Analysis",
                    "description": "Analyze competitive landscape and ecosystem positioning",
                    "requires_rag": True,
                    "with_context": True
                },
                {
                    "chat_type": "SUMMARY",
                    "display_name": "Summary",
                    "description": "Concise 2-page executive summary of key findings",
                    "requires_rag": False,
                    "with_context": True
                },
                {
                    "chat_type": "BOARD_MEMO",
                    "display_name": "Board Memo",
                    "description": "Formal board briefing document with evidence-based analysis",
                    "requires_rag": True,
                    "with_context": True
                }
            ]

single_analysis_service = SingleAnalysisService()