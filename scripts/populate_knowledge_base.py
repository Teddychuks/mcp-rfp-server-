#!/usr/bin/env python3
"""
Populate Knowledge Base Script for MCP-RFP Server
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_rfp_server.config import ServerConfig
from mcp_rfp_server.utils.vector_store import VectorStore
from mcp_rfp_server.tools.knowledge_search import KnowledgeSearcher
from mcp_rfp_server.utils.text_processing import TextProcessor

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def populate_knowledge_base(vector_store: VectorStore, config: ServerConfig):
    """Populate ChromaDB with knowledge base files using intelligent chunking."""
    print("Starting Knowledge Base Population...")

    try:
        kb_path = config.get_knowledge_base_path()
        if not kb_path.exists():
            print(f"Knowledge base directory not found: {kb_path}")
            return False

        txt_files = list(kb_path.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {kb_path}. Skipping population.")
            return True

        print(f"Found {len(txt_files)} knowledge base files")

        all_chunks, all_metadatas, all_ids = [], [], []
        knowledge_searcher = KnowledgeSearcher(config)
        text_processor = TextProcessor()
        await text_processor.initialize()

        for txt_file in txt_files:
            print(f"Processing {txt_file.name}...")
            try:
                content = txt_file.read_text(encoding='utf-8').strip()
                if not content:
                    print(f"   Skipping empty file.")
                    continue

                # --- Improved Sentence-Based Chunking ---
                sentences = text_processor.extract_sentences(content)
                base_category = knowledge_searcher._determine_category(txt_file.name)

                # Create overlapping chunks of sentences
                chunk_size = 3  # Number of sentences per chunk
                overlap = 1  # Number of sentences to overlap

                for i in range(0, len(sentences), chunk_size - overlap):
                    chunk_sentences = sentences[i: i + chunk_size]
                    if not chunk_sentences:
                        continue

                    chunk_text = " ".join(chunk_sentences)
                    chunk_id = f"{txt_file.stem}_{i}"

                    metadata = {
                        "source": txt_file.name,
                        "category": base_category,
                        "chunk_index": str(i),
                        "document_id": txt_file.stem
                    }
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

            except Exception as e:
                print(f"   Error processing {txt_file.name}: {e}")
                continue

        if not all_chunks:
            print("No valid content to add to the knowledge base.")
            return False

        collection_name = "knowledge_base"
        print(f"Cleaning existing collection: {collection_name}")
        await vector_store.delete_collection(collection_name)

        print(f"Adding {len(all_chunks)} semantic chunks to the vector store...")
        result = await vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )

        if result.get("success"):
            print(f"Successfully populated knowledge base!")
            print(f"   Chunks created: {result.get('documents_added')}")

            stats = await vector_store.get_collection_stats(collection_name)
            if stats.get("success"):
                print(f"\nCollection Stats:")
                print(f"   Total chunks: {stats.get('total_chunks')}")
                categories = stats.get('categories', {})
                if categories:
                    print(f"   Categories: {', '.join(categories.keys())}")
                    for category, count in categories.items():
                        print(f"      - {category}: {count} chunks")
            return True
        else:
            print(f"Failed to populate knowledge base: {result.get('error')}")
            return False

    except Exception as e:
        logging.error(f"Unexpected error during population: {e}", exc_info=True)
        return False


async def test_knowledge_base(vector_store: VectorStore):
    """Test the populated knowledge base"""
    print("\nTesting Knowledge Base...")

    test_queries = [
        "company overview and mission",
        "security compliance and certifications",
        "past performance and project experience",
        "technical capabilities and platforms"
    ]

    for query in test_queries:
        print(f"\n   Testing query: '{query}'")
        results = await vector_store.semantic_search(
            collection_name="knowledge_base",
            query=query,
            n_results=2
        )

        if results.get("success"):
            print(f"   Found {len(results.get('results', []))} results:")
            for i, result in enumerate(results.get('results', []), 1):
                score = result.get('similarity_score', 0)
                source = result.get('metadata', {}).get('source', 'unknown')
                print(f"     {i}. {source} (similarity: {score:.3f})")
        else:
            print(f"   Search failed: {results.get('error')}")


async def main():
    """Main execution function"""
    config = ServerConfig.from_env()
    vector_store = VectorStore(
        db_path=str(config.get_chroma_db_path()),
        embedding_model=config.embedding_model
    )
    await vector_store.initialize()
    print(f"Vector store initialized with {config.embedding_model}")

    success = await populate_knowledge_base(vector_store, config)

    if success:
        await test_knowledge_base(vector_store)
        print("\nKnowledge base population and testing completed successfully!")
    else:
        print("\nKnowledge base population failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)