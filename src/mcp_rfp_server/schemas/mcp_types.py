"""
MCP Protocol Types for RFP Server
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class MCPContentType(str, Enum):
    """MCP content types"""
    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"


class MCPContent(BaseModel):
    """Base MCP content"""
    type: MCPContentType


class MCPTextContent(MCPContent):
    """Text content for MCP"""
    type: MCPContentType = MCPContentType.TEXT
    text: str


class MCPImageContent(MCPContent):
    """Image content for MCP"""
    type: MCPContentType = MCPContentType.IMAGE
    data: str = Field(description="Base64 encoded image data")
    mimeType: str


class MCPResourceContent(MCPContent):
    """Resource content for MCP"""
    type: MCPContentType = MCPContentType.RESOURCE
    resource: Dict[str, Any]


class MCPToolCall(BaseModel):
    """MCP tool call request"""
    name: str = Field(description="Tool name")
    arguments: Dict[str, Any] = Field(description="Tool arguments")


class MCPToolResult(BaseModel):
    """MCP tool call result"""
    content: List[Union[MCPTextContent, MCPImageContent, MCPResourceContent]]
    isError: Optional[bool] = Field(default=False)


class MCPResource(BaseModel):
    """MCP resource definition"""
    uri: str = Field(description="Resource URI")
    name: str = Field(description="Human-readable name")
    description: Optional[str] = Field(default=None)
    mimeType: Optional[str] = Field(default=None)


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    inputSchema: Dict[str, Any] = Field(description="JSON schema for input")


class MCPServerCapabilities(BaseModel):
    """MCP server capabilities"""
    tools: Optional[Dict[str, Any]] = Field(default_factory=dict)
    resources: Optional[Dict[str, Any]] = Field(default_factory=dict)
    prompts: Optional[Dict[str, Any]] = Field(default_factory=dict)
    logging: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MCPClientInfo(BaseModel):
    """MCP client information"""
    name: str
    version: str


class MCPServerInfo(BaseModel):
    """MCP server information"""
    name: str
    version: str
    protocolVersion: str = "2024-11-05"
    capabilities: MCPServerCapabilities


class MCPInitializeParams(BaseModel):
    """MCP initialization parameters"""
    protocolVersion: str
    capabilities: Dict[str, Any]
    clientInfo: MCPClientInfo


class MCPInitializeResult(BaseModel):
    """MCP initialization result"""
    protocolVersion: str
    capabilities: MCPServerCapabilities
    serverInfo: MCPServerInfo


class MCPError(BaseModel):
    """MCP error response"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class MCPRequest(BaseModel):
    """MCP JSON-RPC request"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC response"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None


class MCPNotification(BaseModel):
    """MCP JSON-RPC notification"""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None