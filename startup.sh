#!/bin/bash

# Exit immediately if a command exits with a non-zero status.co
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

# Check knowledge source configuration
KNOWLEDGE_SOURCE="${KNOWLEDGE_SOURCE:-local}"
echo "Knowledge source configured as: $KNOWLEDGE_SOURCE"

if [ "$KNOWLEDGE_SOURCE" = "sharepoint" ]; then
    echo "Validating SharePoint configuration..."

    # Validate SharePoint credentials
    if [ -z "$SHAREPOINT_TENANT_ID" ]; then
        echo "FATAL: SHAREPOINT_TENANT_ID is required when KNOWLEDGE_SOURCE=sharepoint"
        exit 1
    fi

    if [ -z "$SHAREPOINT_CLIENT_ID" ]; then
        echo "FATAL: SHAREPOINT_CLIENT_ID is required when KNOWLEDGE_SOURCE=sharepoint"
        exit 1
    fi

    if [ -z "$SHAREPOINT_CLIENT_SECRET" ]; then
        echo "FATAL: SHAREPOINT_CLIENT_SECRET is required when KNOWLEDGE_SOURCE=sharepoint"
        exit 1
    fi

    if [ -z "$SHAREPOINT_SITE_URL" ]; then
        echo "FATAL: SHAREPOINT_SITE_URL is required when KNOWLEDGE_SOURCE=sharepoint"
        exit 1
    fi

    echo "SharePoint configuration validated."
    echo "SharePoint Site: $SHAREPOINT_SITE_URL"
    echo "SharePoint Folder: ${SHAREPOINT_FOLDER_PATH:-knowledge_base}"

elif [ "$KNOWLEDGE_SOURCE" = "local" ]; then
    echo "Validating local knowledge base..."

    KB_PATH="${KNOWLEDGE_BASE_PATH:-/app/knowledge_base}"
    if [ ! -d "$KB_PATH" ] || [ -z "$(ls -A "$KB_PATH")" ]; then
        echo "FATAL: Local knowledge base directory is missing or empty at $KB_PATH."
        echo "Either add .txt files to the directory or switch to SharePoint mode."
        exit 1
    fi

    echo "Local knowledge base validated at: $KB_PATH"

else
    echo "FATAL: Invalid KNOWLEDGE_SOURCE '$KNOWLEDGE_SOURCE'. Must be 'local' or 'sharepoint'."
    exit 1
fi

echo "Environment checks passed."

# --- Knowledge Base Population ---
echo "2. Populating knowledge base..."
echo "This will sync from $KNOWLEDGE_SOURCE source..."

# Run the populate script (it will automatically detect source from config)
python -m scripts.populate_knowledge_base

if [ $? -ne 0 ]; then
    echo "FATAL: Knowledge base population failed. Server will not start."
    exit 1
fi

echo "Knowledge base populated successfully from $KNOWLEDGE_SOURCE."

# --- Configuration Validation ---
echo "3. Validating server configuration..."
python -m src.mcp_rfp_server.main --validate-config
echo "Configuration validation passed."

# --- Start Server ---
echo "4. Starting MCP-RFP Server on port 8000..."
echo "Knowledge source: $KNOWLEDGE_SOURCE"

if [ "$KNOWLEDGE_SOURCE" = "sharepoint" ]; then
    echo "SharePoint integration active - knowledge base will be synced from SharePoint on each restart"
else
    echo "Local files mode - using knowledge base files from mounted volume"
fi

exec python -m src.mcp_rfp_server.main --transport tcp --host 0.0.0.0 --port 8000