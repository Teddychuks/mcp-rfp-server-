"""
Knowledge Search Tool for MCP-RFP Server
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..config import ServerConfig
from ..schemas.rfp_schemas import SearchResult

logger = logging.getLogger(__name__)


class KnowledgeSearcher:
    """Handles knowledge base search operations using ChromaDB"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False

    async def initialize(self):
        """Initialize ChromaDB and embedding model"""
        if self._initialized:
            return

        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.embedding_model)

            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.config.get_chroma_db_path()),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("knowledge_base")
            except Exception:
                logger.info("Creating new knowledge base collection")
                self.collection = self.chroma_client.create_collection("knowledge_base")
                await self._populate_knowledge_base()

            self._initialized = True
            logger.info("Knowledge searcher initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge searcher: {str(e)}")
            raise

    async def search(
            self,
            query: str,
            categories: List[str] = None,
            limit: int = 5
    ) -> Dict[str, Any]:
        """Search knowledge base for relevant information"""

        await self.initialize()

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Build metadata filter
            where_clause = None
            if categories:
                where_clause = {"category": {"$in": categories}}

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit, self.config.max_search_results),
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            search_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 1.0

                    # Convert distance to relevance score (0-1, higher is better)
                    relevance_score = max(0.0, 1.0 - distance)

                    # Apply similarity threshold
                    # NOTE: Lowered threshold for better results on general queries
                    if relevance_score < 0.5:
                        continue

                    search_result = SearchResult(
                        content=doc,
                        metadata=metadata,
                        relevance_score=relevance_score,
                        source=metadata.get("source", "unknown"),
                        category=metadata.get("category", "general")
                    )
                    search_results.append(search_result)

            # Sort by relevance score
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)

            return {
                "success": True,
                "query": query,
                "results": [result.dict() for result in search_results],
                "total_found": len(search_results),
                "categories_searched": categories or [],
                "search_metadata": {
                    "embedding_model": self.config.embedding_model,
                    "collection_size": self.collection.count(),
                    "similarity_threshold": 0.5
                }
            }

        except Exception as e:
            logger.error(f"Knowledge search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "total_found": 0
            }

    async def search_similar_content(
            self,
            reference_text: str,
            similarity_threshold: float = None,
            limit: int = 10
    ) -> Dict[str, Any]:
        """Find content similar to reference text"""

        await self.initialize()

        threshold = similarity_threshold or self.config.similarity_threshold

        try:
            # Generate embedding for reference text
            ref_embedding = self.embedding_model.encode(reference_text).tolist()

            # Search for similar content
            results = self.collection.query(
                query_embeddings=[ref_embedding],
                n_results=limit * 2,  # Get extra to filter by threshold
                include=["documents", "metadatas", "distances"]
            )

            similar_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance

                    # Apply similarity threshold
                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                        similar_results.append({
                            "content": doc,
                            "similarity_score": similarity,
                            "metadata": metadata,
                            "source": metadata.get("source", "unknown"),
                            "category": metadata.get("category", "general")
                        })

            # Sort by similarity and limit results
            similar_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            similar_results = similar_results[:limit]

            return {
                "success": True,
                "reference_text_preview": reference_text[:200] + "..." if len(reference_text) > 200 else reference_text,
                "similarity_threshold": threshold,
                "similar_content": similar_results,
                "total_found": len(similar_results)
            }

        except Exception as e:
            logger.error(f"Similar content search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "similar_content": [],
                "total_found": 0
            }

    async def get_knowledge_categories(self) -> Dict[str, Any]:
        """Get available knowledge base categories and statistics"""

        await self.initialize()

        try:
            # Get all documents
            all_results = self.collection.get(include=["metadatas"])

            categories = {}
            sources = {}

            if all_results and all_results.get("metadatas"):
                for metadata in all_results["metadatas"]:
                    # Count categories
                    category = metadata.get("category", "uncategorized")
                    categories[category] = categories.get(category, 0) + 1

                    # Count sources
                    source = metadata.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1

            return {
                "success": True,
                "total_documents": self.collection.count(),
                "categories": categories,
                "sources": sources,
                "available_categories": list(categories.keys()),
                "available_sources": list(sources.keys())
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge categories: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "categories": {},
                "sources": {}
            }

    async def _populate_knowledge_base(self):
        """Populate knowledge base from files"""

        knowledge_base_path = self.config.get_knowledge_base_path()

        if not knowledge_base_path.exists():
            logger.warning(f"Knowledge base directory not found: {knowledge_base_path}")
            return

        documents = []
        metadatas = []
        ids = []

        # Process all .txt files in knowledge base directory
        txt_files = list(knowledge_base_path.glob("*.txt"))

        if not txt_files:
            logger.warning("No .txt files found in knowledge base directory")
            return

        logger.info(f"Processing {len(txt_files)} knowledge base files...")

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if not content:
                    continue

                # Split content into chunks (paragraphs)
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

                for i, chunk in enumerate(chunks):
                    if len(chunk) < 50:  # Skip very short chunks
                        continue

                    # Determine category from filename
                    category = self._determine_category(txt_file.name)

                    documents.append(chunk)
                    metadatas.append({
                        "source": txt_file.name,
                        "category": category,
                        "chunk_index": i,
                        "file_path": str(txt_file)
                    })
                    ids.append(f"{txt_file.stem}_{i}")

            except Exception as e:
                logger.error(f"Error processing file {txt_file}: {str(e)}")
                continue

        if not documents:
            logger.warning("No documents to add to knowledge base")
            return

        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} document chunks...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=False)

            # Add to ChromaDB
            logger.info("Adding documents to ChromaDB...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully populated knowledge base with {len(documents)} document chunks")

        except Exception as e:
            logger.error(f"Failed to populate knowledge base: {str(e)}")
            raise

    def _determine_category(self, filename: str) -> str:
        """Determine category based on filename"""

        filename_lower = filename.lower()

        if "company" in filename_lower or "overview" in filename_lower:
            return "company_overview"
        elif "performance" in filename_lower or "past" in filename_lower:
            return "past_performance"
        elif "security" in filename_lower or "compliance" in filename_lower:
            return "security_compliance"
        elif "technical" in filename_lower or "product" in filename_lower:
            return "technical_capabilities"
        elif "management" in filename_lower or "approach" in filename_lower:
            return "management_approach"
        elif "quality" in filename_lower or "assurance" in filename_lower:
            return "quality_assurance"
        elif "personnel" in filename_lower or "bio" in filename_lower:
            return "personnel_bios"
        else:
            return "general"

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the knowledge base"""

        await self.initialize()

        try:
            stats = await self.get_knowledge_categories()

            if stats["success"]:
                stats["embedding_model"] = self.config.embedding_model
                stats["vector_dimensions"] = 384  # MiniLM dimensions
                stats["last_updated"] = "startup"  # Would track actual updates

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }