"""
MCP-RFP Server - Unified MCP Server with Gemini Integration
"""
import asyncio
import logging
import sys
import argparse

# Correctly import the server class from its own package
from .mcp.server import MCPRFPServer
from .config import ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="MCP-RFP Server")
    parser.add_argument(
        '--transport', type=str, default='tcp', choices=['stdio', 'tcp'],
        help='The transport protocol to use (tcp or stdio).'
    )
    parser.add_argument(
        '--host', type=str, default='127.0.0.1',
        help='The host to bind to for TCP transport.'
    )
    parser.add_argument(
        '--port', type=int, default=8000,
        help='The port to listen on for TCP transport.'
    )
    parser.add_argument(
        '--validate-config', action='store_true',
        help='Validate the configuration and exit.'
    )
    args = parser.parse_args()

    try:
        if args.validate_config:
            config = ServerConfig.from_env()
            config.ensure_directories()
            config.validate_configuration()
            logger.info("Configuration validation successful")
            sys.exit(0)

        config = ServerConfig.from_env()
        config.ensure_directories()

        server = MCPRFPServer(config)

        logger.info("Starting MCP-RFP Server...")
        logger.info(f"Server name: {config.server_name}")
        logger.info(f"Transport: {args.transport}")

        if args.transport == 'tcp':
            await server.run_tcp_server(host=args.host, port=args.port)
        else:
            logger.info("MCP-RFP Server is ready and listening on stdio...")
            await server.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("MCP-RFP Server stopped")


if __name__ == "__main__":
    asyncio.run(main())