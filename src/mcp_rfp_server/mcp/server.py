"""
Unified MCP Server with Gemini Integration - Business Logic Layer
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Callable, Awaitable

from .protocol_handler import MCPProtocolHandler
from ..config import ServerConfig
from ..gemini.orchestrator import GeminiOrchestrator
from ..resources.knowledge_base import KnowledgeBaseResource
from ..resources.templates import TemplateResource
from ..tools.document_processor import DocumentProcessor
from ..tools.knowledge_search import KnowledgeSearcher
from ..tools.proposal_generator import ProposalGenerator

logger = logging.getLogger(__name__)

_tools_registry: Dict[str, Dict[str, Any]] = {}

def tool(name: str, description: str, input_schema: Dict[str, Any]):
    """Decorator to register a method as an MCP tool."""
    def decorator(func: Callable[..., Awaitable[Dict[str, Any]]]):
        _tools_registry[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "function": func
        }
        return func
    return decorator

class MCPRFPServer:
    """Unified MCP Server with integrated Gemini orchestration"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.protocol_handler = MCPProtocolHandler(config, self)

        # Initialize components
        self.knowledge_searcher = KnowledgeSearcher(config)
        self.gemini_orchestrator = GeminiOrchestrator(config, None, self.knowledge_searcher)
        self.document_processor = DocumentProcessor(config, self.gemini_orchestrator)
        self.gemini_orchestrator.document_processor = self.document_processor

        self.proposal_generator = ProposalGenerator(config, self.gemini_orchestrator)
        self.gemini_orchestrator.proposal_generator = self.proposal_generator
        self.knowledge_base = KnowledgeBaseResource(config)
        self.templates = TemplateResource(config)

    async def _dispatch_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Central dispatcher for all incoming MCP requests."""
        if method == "initialize":
            return await self.protocol_handler._handle_initialize(params)
        elif method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            return await self.call_tool(params.get("name"), params.get("arguments"))
        elif method == "resources/list":
            return await self.list_resources()
        elif method == "resources/read":
            return await self.read_resource(params.get("uri"))
        else:
            raise ValueError(f"Method not found: {method}")

    async def list_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return available tools from the registry."""
        tools_list = [
            {k: v for k, v in tool_def.items() if k != 'function'}
            for tool_def in _tools_registry.values()
        ]
        return {"tools": tools_list}

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically call a tool from the registry."""
        if not name or name not in _tools_registry:
            raise ValueError(f"Unknown tool: {name}")

        tool_def = _tools_registry[name]
        result = await tool_def["function"](self, **(arguments or {}))
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    # --- Tool Definitions ---

    @tool(name="process_rfp_document", description="Complete RFP processing workflow", input_schema={})
    async def process_rfp_document(self, **kwargs) -> Dict[str, Any]:
        return await self.gemini_orchestrator.process_rfp_document(**kwargs)

    @tool(name="extract_rfp_requirements", description="Extract requirements from RFP documents", input_schema={})
    async def extract_rfp_requirements(self, **kwargs) -> Dict[str, Any]:
        return await self.document_processor.extract_requirements(**kwargs)

    @tool(name="search_knowledge_base", description="Search knowledge base", input_schema={})
    async def search_knowledge_base(self, **kwargs) -> Dict[str, Any]:
        return await self.knowledge_searcher.search(**kwargs)

    @tool(name="generate_proposal_section", description="Generate proposal response", input_schema={})
    async def generate_proposal_section(self, **kwargs) -> Dict[str, Any]:
        return await self.proposal_generator.generate_with_gemini(**kwargs)

    @tool(name="analyze_requirements", description="Analyze requirements with Gemini", input_schema={})
    async def analyze_requirements(self, **kwargs) -> Dict[str, Any]:
        return await self.gemini_orchestrator.analyze_requirements(**kwargs)

    # --- End Tool Definitions ---

    async def list_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        kb_resources = await self.knowledge_base.list_resources()
        template_resources = await self.templates.list_resources()
        all_resources = [res.model_dump() for res in (kb_resources + template_resources)]
        return {"resources": all_resources}

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        if uri.startswith("knowledge://"):
            return await self.knowledge_base.read_resource(uri)
        elif uri.startswith("template://"):
            return await self.templates.read_resource(uri)
        else:
            raise ValueError(f"Unknown resource URI scheme: {uri}")

    async def initialize(self):
        logger.info("Initializing MCP-RFP Server components...")
        await asyncio.gather(
            self.gemini_orchestrator.initialize(), self.document_processor.initialize(),
            self.knowledge_searcher.initialize(), self.proposal_generator.initialize(),
            self.knowledge_base.initialize(), self.templates.initialize()
        )
        logger.info("All components initialized successfully")

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Main loop to handle a single client connection."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected from {addr}")
        buffer = b""
        try:
            while True:
                chunk = await reader.read(4096)  # Read in 4KB chunks
                if not chunk:
                    break  # Connection closed by client

                buffer += chunk

                # Process all complete messages in the buffer
                while b'\n' in buffer:
                    line_bytes, buffer = buffer.split(b'\n', 1)
                    line = line_bytes.decode('utf-8')

                    response_str = await self.protocol_handler.handle_message(line)
                    if response_str:
                        writer.write(response_str.encode('utf-8') + b'\n')
                        await writer.drain()

        except ConnectionResetError:
            logger.warning(f"Client {addr} disconnected abruptly.")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            logger.info(f"Closing connection from {addr}")
            writer.close()
            await writer.wait_closed()

    async def run(self):
        await self.initialize()
        logger.info("MCP server components initialized, starting stdio transport")
        await self.handle_connection(asyncio.get_event_loop()._reader, asyncio.get_event_loop()._writer)

    async def run_tcp_server(self, host: str, port: int):
        await self.initialize()
        server = await asyncio.start_server(self.handle_connection, host, port)
        logger.info(f"MCP server listening on {host}:{port}")
        async with server:
            await server.serve_forever()