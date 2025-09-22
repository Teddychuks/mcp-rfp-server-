#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting MCP-RFP Server ---"

# Set Python path to ensure src is importable
export PYTHONPATH=/app

# --- Pre-flight Checks ---
echo "1. Verifying environment..."
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "FATAL: GOOGLE_API_KEY environment variable is not set."
    exit 1
fi

KB_PATH="/app/knowledge_base"
if [ ! -d "$KB_PATH" ] || [ -z "$(ls -A "$KB_PATH")" ]; then
    echo "FATAL: Knowledge base directory is missing or empty at $KB_PATH."
    exit 1
fi
echo "Environment checks passed."

# --- Knowledge Base Population ---
echo "2. Populating knowledge base from files..."
# This command now runs inside the container on every start
python -m scripts.populate_knowledge_base

if [ $? -ne 0 ]; then
    echo "FATAL: Knowledge base population failed. Server will not start."
    exit 1
fi
echo "Knowledge base populated successfully."


# --- Configuration Validation ---
echo "3. Validating server configuration..."
python -m src.mcp_rfp_server.main --validate-config
echo "Configuration validation passed."

# --- Start Server ---
echo "4. Starting MCP-RFP Server on port 8000..."
exec python -m src.mcp_rfp_server.main --transport tcp --host 0.0.0.0 --port 8000