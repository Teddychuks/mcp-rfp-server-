"""
Vector Store Utilities for MCP-RFP Server
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np

from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class VectorStore:
    """Advanced vector store operations for ChromaDB"""

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.chroma_client = None
        self.text_processor = TextProcessor()
        self._initialized = False

    async def initialize(self):
        """Initialize vector store components"""
        if self._initialized:
            return

        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                # Disable telemetry to prevent connection errors
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Initialize text processor
            await self.text_processor.initialize()

            self._initialized = True
            logger.info(f"Vector store initialized with {self.embedding_model_name}")

        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise

    async def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any] = None):
        """Get existing collection or create a new one with default metadata."""
        await self.initialize()
        # Add default metadata to satisfy ChromaDB's requirement for new collections.
        if metadata is None:
            metadata = {"description": "Knowledge base for mcp-rfp-server"}
        return self.chroma_client.get_or_create_collection(name=collection_name, metadata=metadata)

    async def add_documents(
            self,
            collection_name: str,
            documents: List[str],
            metadatas: List[Dict[str, Any]],
            ids: List[str]
    ) -> Dict[str, Any]:
        """Add documents to a collection."""
        await self.initialize()
        try:
            collection = await self.get_or_create_collection(collection_name)

            # Generate embeddings for the provided documents/chunks
            embeddings = self.embedding_model.encode(documents, show_progress_bar=False).tolist()

            # Add to ChromaDB in batches to avoid potential issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    documents=documents[i:i+batch_size]
                )

            return {
                "success": True,
                "documents_added": len(documents),
                "collection": collection_name
            }
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


    async def semantic_search(
            self,
            collection_name: str,
            query: str,
            n_results: int = 5,
            where: Dict[str, Any] = None,
            score_threshold: float = None
    ) -> Dict[str, Any]:
        """Perform semantic search"""
        await self.initialize()

        try:
            collection = self.chroma_client.get_collection(collection_name)

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            processed_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # Convert distance to similarity

                    # Apply score threshold if specified
                    if score_threshold and similarity < score_threshold:
                        continue

                    processed_results.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity,
                        "distance": distance
                    })

            return {
                "success": True,
                "query": query,
                "results": processed_results,
                "total_found": len(processed_results)
            }

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        await self.initialize()

        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()

            if count == 0:
                 return {
                    "success": True, "total_chunks": 0, "categories": {},
                 }


            # Get all documents for analysis
            all_data = collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", [])

            categories = {}
            for metadata in metadatas:
                category = metadata.get("category", "uncategorized")
                categories[category] = categories.get(category, 0) + 1


            return {
                "success": True,
                "collection_name": collection_name,
                "total_chunks": count,
                "categories": categories
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"success": False, "error": str(e)}

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection if it exists."""
        await self.initialize()
        try:
            # This is more robust than list_collections()
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except ValueError:
            logger.info(f"Collection '{collection_name}' does not exist, skipping deletion.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}", exc_info=True)
            return False