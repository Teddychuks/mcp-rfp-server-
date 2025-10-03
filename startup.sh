#!/bin/bash
set -e
echo "--- Starting MCP-RFP Server ---"
export PYTHONPATH=/app/src

echo "--- Populating Knowledge Base ---"
python /app/scripts/populate_knowledge_base.py
if [ $? -ne 0 ]; then
    echo "FATAL: Knowledge base population failed. Server will not start."
    exit 1
fi
echo "--- Knowledge Base Population Complete ---"

echo "--- Starting Main Server on port 8000 ---"
exec python -m mcp_rfp_server.main --transport tcp --host 0.0.0.0 --port 8000