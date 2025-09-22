"""
Data schemas for MCP-RFP Server
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """RFP requirement data model"""
    id: str = Field(description="Unique requirement identifier")
    text: str = Field(description="Full requirement text")
    type: str = Field(description="Requirement category (security, technical, etc.)")
    priority: str = Field(description="Priority level (high, medium, low)")
    section: str = Field(description="Document section where requirement was found")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResult(BaseModel):
    """Knowledge base search result"""
    content: str = Field(description="Content text")
    metadata: Dict[str, Any] = Field(description="Content metadata")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    source: str = Field(description="Source file or identifier")
    category: str = Field(description="Content category")


class ProposalSection(BaseModel):
    """Generated proposal section"""
    requirement_id: str = Field(description="Associated requirement ID")
    requirement_text: str = Field(description="Original requirement text")
    response: str = Field(description="Generated response text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class DocumentMetadata(BaseModel):
    """Document processing metadata"""
    document_type: str = Field(description="Document format (pdf, docx)")
    word_count: int = Field(description="Total word count")
    character_count: int = Field(description="Total character count")
    sentence_count: int = Field(description="Total sentence count")
    processing_model: str = Field(description="NLP model used")
    extraction_method: str = Field(description="Extraction method used")
    confidence_threshold: float = Field(description="Confidence threshold applied")


class ProcessingPlan(BaseModel):
    """AI-generated processing plan"""
    plan_id: str = Field(description="Unique plan identifier")
    steps: List[Dict[str, Any]] = Field(description="Processing steps")
    success_criteria: List[str] = Field(description="Success criteria")
    estimated_duration: str = Field(description="Estimated processing time")


class RequirementAnalysis(BaseModel):
    """Requirement analysis results"""
    analysis_type: str = Field(description="Type of analysis performed")
    requirements_count: int = Field(description="Number of requirements analyzed")
    priority_ranking: List[str] = Field(default_factory=list, description="Priority order")
    recommendations: List[str] = Field(default_factory=list, description="Analysis recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")


class MCPToolResult(BaseModel):
    """Standard result format for MCP tools"""
    success: bool = Field(description="Operation success status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")


class ServerStatus(BaseModel):
    """Server status information"""
    server_name: str = Field(description="Server name")
    version: str = Field(description="Server version")
    status: str = Field(description="Current status")
    uptime: str = Field(description="Server uptime")
    components: Dict[str, bool] = Field(description="Component status")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")