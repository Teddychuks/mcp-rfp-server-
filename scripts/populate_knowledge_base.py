#!/usr/bin/env python3
"""
Populate Knowledge Base Script for MCP-RFP Server with SharePoint Integration
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

# SharePoint integration
from mcp_rfp_server.integrations.sharepoint import SharePointClient, SharePointKnowledgeManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def populate_from_local_files(vector_store: VectorStore, config: ServerConfig):
    """Populate ChromaDB with local knowledge base files using intelligent chunking."""
    print("Starting Local Knowledge Base Population...")

    try:
        kb_path = config.get_knowledge_base_path()
        if not kb_path.exists():
            print(f"Knowledge base directory not found: {kb_path}")
            return False

        txt_files = list(kb_path.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {kb_path}. Skipping population.")
            return True

        print(f"Found {len(txt_files)} local knowledge base files")

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

                # Process file content
                chunks, metadatas, ids = await process_file_content(
                    txt_file.name, content, knowledge_searcher, text_processor
                )

                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)

            except Exception as e:
                print(f"   Error processing {txt_file.name}: {e}")
                continue

        return await store_documents(vector_store, all_chunks, all_metadatas, all_ids)

    except Exception as e:
        logging.error(f"Local knowledge base population failed: {e}", exc_info=True)
        return False


async def populate_from_sharepoint(vector_store: VectorStore, config: ServerConfig):
    """Populate ChromaDB with SharePoint knowledge base files."""
    print("Starting SharePoint Knowledge Base Population...")

    try:
        sharepoint_config = config.get_sharepoint_config()

        # Extract folder_path before passing to SharePointClient
        folder_path = sharepoint_config["folder_path"]
        client_config = {k: v for k, v in sharepoint_config.items() if k != "folder_path"}

        print("Connecting to SharePoint...")
        async with SharePointClient(**client_config) as sp_client:
            sp_manager = SharePointKnowledgeManager(sp_client, folder_path)

            print("Syncing files from SharePoint...")
            sync_result = await sp_manager.sync_knowledge_base()

            if not sync_result['success']:
                print(f"SharePoint sync failed: {sync_result['error']}")
                return False

            synced_files = sync_result['synced_files']
            if not synced_files:
                print("No files found in SharePoint knowledge base folder.")
                return False

            print(f"Successfully synced {len(synced_files)} files from SharePoint")

            if sync_result['failed_files']:
                print(f"Warning: {len(sync_result['failed_files'])} files failed to sync:")
                for failed in sync_result['failed_files']:
                    print(f"   - {failed['name']}: {failed['error']}")

            # Process synced files
            all_chunks, all_metadatas, all_ids = [], [], []
            knowledge_searcher = KnowledgeSearcher(config)
            text_processor = TextProcessor()
            await text_processor.initialize()

            for filename in synced_files:
                print(f"Processing {filename}...")
                try:
                    content = sp_manager.get_file_content(filename)
                    if not content or not content.strip():
                        print(f"   Skipping empty file.")
                        continue

                    # Process file content
                    chunks, metadatas, ids = await process_file_content(
                        filename, content, knowledge_searcher, text_processor, is_sharepoint=True
                    )

                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    all_ids.extend(ids)

                except Exception as e:
                    print(f"   Error processing {filename}: {e}")
                    continue

            return await store_documents(vector_store, all_chunks, all_metadatas, all_ids)

    except Exception as e:
        logging.error(f"SharePoint knowledge base population failed: {e}", exc_info=True)
        return False


async def process_file_content(filename: str, content: str, knowledge_searcher, text_processor, is_sharepoint=False):
    """Process a single file's content into chunks."""
    chunks, metadatas, ids = [], [], []

    # Improved Sentence-Based Chunking
    sentences = text_processor.extract_sentences(content)
    base_category = knowledge_searcher._determine_category(filename)

    # Create overlapping chunks of sentences
    chunk_size = 3  # Number of sentences per chunk
    overlap = 1  # Number of sentences to overlap

    for i in range(0, len(sentences), chunk_size - overlap):
        chunk_sentences = sentences[i: i + chunk_size]
        if not chunk_sentences:
            continue

        chunk_text = " ".join(chunk_sentences)
        file_stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
        chunk_id = f"{file_stem}_{i}"

        metadata = {
            "source": filename,
            "category": base_category,
            "chunk_index": str(i),
            "document_id": file_stem,
            "source_type": "sharepoint" if is_sharepoint else "local"
        }

        chunks.append(chunk_text)
        metadatas.append(metadata)
        ids.append(chunk_id)

    return chunks, metadatas, ids


async def store_documents(vector_store: VectorStore, all_chunks, all_metadatas, all_ids):
    """Store processed documents in vector store."""
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


async def populate_knowledge_base(vector_store: VectorStore, config: ServerConfig):
    """Populate ChromaDB based on configured knowledge source."""
    if config.is_sharepoint_enabled():
        print("SharePoint integration enabled")
        return await populate_from_sharepoint(vector_store, config)
    else:
        print("Using local knowledge base files")
        return await populate_from_local_files(vector_store, config)


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
                source_type = result.get('metadata', {}).get('source_type', 'unknown')
                print(f"     {i}. {source} [{source_type}] (similarity: {score:.3f})")
        else:
            print(f"   Search failed: {results.get('error')}")


async def main():
    """Main execution function"""
    config = ServerConfig.from_env()

    # Validate configuration
    try:
        config.validate_configuration()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    config.ensure_directories()

    vector_store = VectorStore(
        db_path=str(config.get_chroma_db_path()),
        embedding_model=config.embedding_model
    )
    await vector_store.initialize()

    print(f"Vector store initialized with {config.embedding_model}")
    print(f"Knowledge source: {config.knowledge_source}")

    if config.is_sharepoint_enabled():
        print(f"SharePoint site: {config.sharepoint_site_url}")
        print(f"SharePoint folder: {config.sharepoint_folder_path}")

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