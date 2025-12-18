from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import List,  Dict, Any, Optional, Union
from datetime import datetime

class AlignmentDetails(BaseModel):
    what_text: str = Field(default="Analysis not available")
    what_aligned: bool = Field(default=False)
    how_text: str = Field(default="Analysis not available") 
    how_aligned: bool = Field(default=False)
    who_text: str = Field(default="Analysis not available")
    who_aligned: bool = Field(default=False)
    why_text: str = Field(default="Analysis not available")

class Evidence(BaseModel):
    what: List[str] = Field(default_factory=list)
    how: List[str] = Field(default_factory=list)
    who: List[str] = Field(default_factory=list)

class VerificationData(BaseModel):
    org_name: str = Field(default="Unknown")
    verified: bool = Field(default=False)
    risk_level: str = Field(default="UNKNOWN")
    ein: Optional[str] = None
    revenue: Optional[float] = 0
    assets: Optional[float] = 0

class ProposalAnalysis(BaseModel):
    filename: str
    organization_name: str = Field(default="Unknown Organization")
    recommendation: str = Field(default="Consider")
    budget: str = Field(default="Not specified")
    timeline: str = Field(default="Not specified") 
    overall_alignment_score: int = Field(default=50, ge=0, le=100)
    alignment: AlignmentDetails = Field(default_factory=AlignmentDetails)
    evidence: Evidence = Field(default_factory=Evidence)
    verification: Optional[VerificationData] = None

class RFPSummary(BaseModel):
    title: str = Field(default="Untitled Analysis")
    executive_summary: str = Field(default="No summary available")
    key_requirements: List[str] = Field(default_factory=list)
    target_population: str = Field(default="Not specified")
    budget_range: str = Field(default="Not specified")
    timeline_expectations: str = Field(default="Not specified")
    evaluation_criteria: List[str] = Field(default_factory=list)

class AnalysisResponse(BaseModel):
    rfp_id: str
    rfp_title: str = Field(default="Analysis Results")
    rfp_summary: RFPSummary = Field(default_factory=RFPSummary)
    proposals: List[ProposalAnalysis] = Field(default_factory=list)
    analysis_type: str = Field(default="instruction_based")


class SingleAnalysisFileRequest(BaseModel):
    """Request model for single analysis with file uploads"""
    files: List[UploadFile] = Field(..., description="Files to analyze")
    chat_type: str = Field(default="ANALYSIS", description="Type of analysis")
    domain: Optional[str] = Field(None, description="Optional domain filter")
    top_k: Optional[int] = Field(10, description="Number of research sources")
    max_tokens: Optional[int] = Field(4000, description="Maximum response length")
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    
class SingleAnalysisRequest(BaseModel):
    """Request model for single analysis"""
    message: str = Field(..., description="User's question or analysis request")
    chat_type: str = Field(default="ANALYSIS", description="Type of analysis (ANALYSIS, BIAS, COUNTERPOINT, etc.)")
    domain: Optional[str] = Field(None, description="Optional domain filter (education, health, etc.)")
    top_k: Optional[int] = Field(10, description="Number of research sources to retrieve (default 10)")
    max_tokens: Optional[int] = Field(4000, description="Maximum response length")

class Citation(BaseModel):
    id: int
    chunk_id: str
    paper_id: str
    filename: str
    section: str
    domain: str
    content: str
    distance: float

class SingleAnalysisResponse(BaseModel):
    """Response model for single analysis"""
    answer: str = Field(..., description="Generated analysis with inline citations")
    citations: List[Citation] = Field(default_factory=list, description="Citation metadata")
    has_research: bool = Field(default=False, description="Whether research was found and used")
    num_sources: int = Field(default=0, description="Number of sources cited")
    chat_type: str = Field(..., description="Type of analysis performed")
    file_metadata: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Uploaded file information"
    )
    content_analysis: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Content statistics"
    )
    analysis_id: Optional[str] = Field(
        None, 
        description="Database ID for this analysis"
    )
    organization_id: Optional[str] = Field(
        None, 
        description="Organization identifier"
    )
    user_id: Optional[str] = Field(
        None, 
        description="User identifier"
    )
    context_used: Optional[bool] = Field(
        False,
        description="Whether initial analysis context was used"
    )
    context_aware: Optional[bool] = Field(
        False,
        description="Whether this was a context-aware analysis"
    )
    analysis_mode: Optional[str] = Field(
        None,
        description="Mode of analysis: 'standard', 'query_based', or 'context_aware'"
    )
    user_query: Optional[str] = Field(
        None,
        description="User's specific question (if query_based mode)"
    )
    created_at: Optional[datetime] = Field(
        None, 
        description="When analysis was created"
    )
    
    class Config:
        from_attributes = True