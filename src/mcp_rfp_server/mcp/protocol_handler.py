"""
MCP Protocol Handler for RFP Server
"""
import json
import logging
from typing import Dict, Any, Optional
from typing import TYPE_CHECKING

from ..config import ServerConfig
from ..schemas.mcp_types import (
    MCPRequest, MCPResponse, MCPNotification, MCPError,
    MCPInitializeParams, MCPInitializeResult, MCPServerInfo, MCPServerCapabilities
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MCPProtocolHandler:
    """Handles MCP JSON-RPC protocol communication"""

    def __init__(self, config: ServerConfig, server_instance: 'MCPRFPServer'):
        self.config = config
        self.server = server_instance
        self.initialized = False
        self.client_info = None
        self.server_info = MCPServerInfo(
            name=config.server_name,
            version=config.server_version,
            protocolVersion=config.mcp_protocol_version,
            capabilities=MCPServerCapabilities()
        )

    async def handle_message(self, message: str) -> Optional[str]:
        """Handle a single incoming MCP message (request or notification)"""
        request_id = None
        try:
            if not message.strip():
                return None

            data = json.loads(message)
            request_id = data.get("id")

            if "id" not in data:
                await self._handle_notification(data)
                return None

            request = MCPRequest(**data)
            response = await self._handle_request(request)
            return json.dumps(response.model_dump(exclude_none=True))

        except json.JSONDecodeError:
            error = MCPError(code=-32700, message="Parse error: Invalid JSON")
            return json.dumps(MCPResponse(id=None, error=error).model_dump(exclude_none=True))
        except Exception as e:
            logger.error(f"Protocol handler error: {e}", exc_info=True)
            error = MCPError(code=-32603, message=f"Internal error: {str(e)}")
            return json.dumps(MCPResponse(id=request_id, error=error).model_dump(exclude_none=True))

    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Route and handle an RPC request"""
        if not self.initialized and request.method != "initialize":
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32002, message="Server not initialized. Call 'initialize' first.")
            )

        try:
            # Delegate the entire request to the main server's dispatcher
            result_data = await self.server._dispatch_request(request.method, request.params or {})
            return MCPResponse(id=request.id, result=result_data)

        except Exception as e:
            logger.error(f"Error handling method {request.method}: {e}", exc_info=True)
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32603, message=f"Internal error during method execution: {str(e)}")
            )

    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle a notification"""
        notification = MCPNotification(**data)
        if notification.method == "notifications/initialized":
            self.initialized = True
            if self.client_info:
                logger.info(
                    f"Client connection initialized: "
                    f"{self.client_info.name} v{self.client_info.version}"
                )
            else:
                logger.info("Client connection initialized.")
        else:
            logger.warning(f"Received unknown notification: {notification.method}")

    async def _handle_initialize(self, params: Dict[str, Any]) -> dict:
        """Handle the initialize request."""
        init_params = MCPInitializeParams.model_validate(params)
        self.client_info = init_params.clientInfo
        logger.info(
            f"Initialization request from client: "
            f"{self.client_info.name} v{self.client_info.version}"
        )
        result = MCPInitializeResult(
            protocolVersion=self.server_info.protocolVersion,
            capabilities=self.server_info.capabilities,
            serverInfo=self.server_info
        )
        return result.model_dump()