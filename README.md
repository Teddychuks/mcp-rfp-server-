Of course. Here is a comprehensive, Docker-specific `README.md` for the MCP RFP Server.

-----

# MCP RFP Server

## Overview

The **MCP RFP Server** is the backend engine for an intelligent Request for Proposal (RFP) response automation system. It leverages AI orchestration, document intelligence, and a Retrieval-Augmented Generation (RAG) pipeline to automate the process of creating proposal documents.

This server is designed to be run as a containerized service using Docker, ensuring a consistent, reproducible, and secure operating environment.

-----

## Deployment with Docker

This is the recommended and standard method for deploying the server. It encapsulates all dependencies, models, and configurations into a portable image.

### **Prerequisites**

* **Docker and Docker Compose**: Must be installed and running on your deployment machine.
* **Google API Key**: A valid `GOOGLE_API_KEY` is required for all AI-driven features, including content generation and intelligent document analysis.

### **1. Configuration**

Before building or running the container, you must provide the necessary environment variables.

Create a file named `.env` in the root directory of the `mcp-rfp-server` project with the following content:

```
# .env file for server configuration

# Required: Your Google AI API Key for generative features.
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

Replace `"YOUR_API_KEY_HERE"` with your actual API key.

### **2. Knowledge Base Setup**

The server's ability to generate accurate, context-aware responses is directly dependent on the quality of the documents you provide.

* Place all your knowledge base documents (e.g., past proposals, company policies, security compliance documents, technical specifications) as `.txt` files inside the `knowledge_base` directory.
* The `startup.sh` script, which runs when the container starts, will automatically find these files, process them into a searchable format, and populate the vector database. This process runs every time the container starts, ensuring the knowledge base is always up-to-date with the provided files.

### **3. Building and Running the Container**

The `docker-compose.yml` file is the primary way to manage the server's lifecycle.

To build the Docker image and start the server, run the following command from the root of the `mcp-rfp-server` project:

```bash
docker-compose up --build
```

This single command automates the entire setup process:

1.  **Builds the Docker image**: It uses the `Dockerfile` to create a secure, non-root image. This process includes installing all Python dependencies and pre-downloading the required machine learning models to speed up startup time.
2.  **Starts the container**: It runs the `startup.sh` script inside the new container. This script performs several critical pre-flight checks:
    * Verifies that the `GOOGLE_API_KEY` is available.
    * Confirms that the `knowledge_base` directory is not empty.
    * Runs the `populate_knowledge_base.py` script to index your documents.
    * Finally, starts the TCP server on port 8000.
3.  **Mounts Volumes**: The `docker-compose.yml` file is configured to mount local directories into the container. This is crucial for data persistence and means that your vector database (`chroma_data`) and any server-generated files (`outputs`) will be saved on your host machine even if the container is stopped or removed.

Once started, the server will be running and listening for network connections on `127.0.0.1:8000`.

### **Stopping the Server**

To stop the server container, press `Ctrl+C` in the terminal where `docker-compose` is running.