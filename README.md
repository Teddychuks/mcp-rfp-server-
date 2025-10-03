# MCP-RFP Server

The MCP-RFP Server is the backend component for an AI-powered system designed to automate the process of responding to Requests for Proposal (RFPs). It uses a combination of large language models (via Gemini API) and a vector knowledge base to analyze RFP documents and generate tailored responses.

## Features

- **AI-Powered Analysis**: Utilizes the Gemini API to understand and break down RFP requirements.
- **Vector Knowledge Base**: Creates a searchable knowledge base from your company's documents using ChromaDB.
- **Multiple Knowledge Sources**: Can load knowledge base documents from local files, SharePoint, or Google Drive.
- **Dockerized**: Easy to set up and run in a consistent environment using Docker.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ and Poetry
- Access to the Google Gemini API

---

## Configuration

The server is configured using a `.env` file in the project root. Create this file before proceeding.

### **Core Configuration**

| Variable        | Description                               | Default                        |
|-----------------|-------------------------------------------|--------------------------------|
| `GOOGLE_API_KEY`  | **Required.** Your API key for Gemini.    | `""`                           |
| `KNOWLEDGE_SOURCE`| The source for the knowledge base. Options: `gdrive`, `sharepoint`, `local`. | `gdrive` |

### **Knowledge Source Options**

Fill in the variables for your chosen `KNOWLEDGE_SOURCE`.

#### 1. Google Drive (`gdrive`)

| Variable                          | Description                                      | Default |
|-----------------------------------|--------------------------------------------------|---------|
| `GDRIVE_KNOWLEDGE_BASE_FOLDER_ID` | **Required.** The ID of the Google Drive folder. | `""`    |

#### 2. SharePoint (`sharepoint`)

| Variable                  | Description                                            | Default          |
|---------------------------|--------------------------------------------------------|------------------|
| `SHAREPOINT_TENANT_ID`    | **Required.** Your Azure Tenant ID.                    | `""`             |
| `SHAREPOINT_CLIENT_ID`    | **Required.** The Application (client) ID.             | `""`             |
| `SHAREPOINT_CLIENT_SECRET`| **Required.** The client secret for the application.   | `""`             |
| `SHAREPOINT_SITE_URL`     | **Required.** The URL of the SharePoint site.          | `""`             |

#### 3. Local Files (`local`)

The server will look for `.txt` files in the `./knowledge_base` directory by default.

---

## Setup and Installation

### **1. Google Drive API Setup (Required for `gdrive` mode)**

If using Google Drive, perform this one-time setup:

1.  **Enable the Google Drive API** in the Google Cloud Console.
2.  **Configure the OAuth Consent Screen** and add your email as a **Test User**.
3.  **Create OAuth 2.0 Credentials** for a **Desktop app**.
4.  Download the credentials file and rename it to **`credentials.json`**.
5.  Place this `credentials.json` file in the root directory of this project.

### **2. Install Local Dependencies**

You must install dependencies locally to run the one-time authentication script.
```bash
poetry install
```

### **3. First-Time Google Drive Authentication (Required for `gdrive` mode)**

Run this command **once** on your local machine. This will open a browser for you to log in and will create a `token.json` file that Docker needs to authenticate without a browser.

```bash
# Make sure your .env file is configured for gdrive
python scripts/populate_knowledge_base.py
```

-----

## Running the Server

### **Using Docker (Recommended)**

This is the most reliable way to run the server.

**For the very first run, or after making code changes:**
Use the `--build` command to create a new Docker image with your latest code.

```bash
# Build the image and start the server
docker-compose up --build
```

**To start the server on subsequent runs:**

```bash
docker-compose up
```

**To stop the server:**
Press `Ctrl + C` in the terminal where the server is running, then run:

```bash
docker-compose down
```

**If you encounter stubborn caching issues:**
Run this sequence to completely clear all old Docker data and rebuild from scratch.

```bash
# Stop and remove any existing containers
docker-compose down

# WARNING: This removes all unused Docker data, including old images and build cache
docker system prune -a

# Build and run fresh
docker-compose up --build
```

### **Local Development (Without Docker)**

This method runs the server directly on your machine. The script will first populate the knowledge base and then start the TCP server.

```bash
# Ensure dependencies are installed with `poetry install`
poetry run python -m mcp_rfp_server.main --transport tcp --host 127.0.0.1 --port 8000
```