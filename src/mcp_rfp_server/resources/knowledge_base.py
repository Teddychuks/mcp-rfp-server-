"""
Knowledge Base Resource for MCP-RFP Server
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from mcp.types import Resource, TextContent, ReadResourceResult

from ..config import ServerConfig

logger = logging.getLogger(__name__)


class KnowledgeBaseResource:
    """Provides MCP resource access to knowledge base content"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._initialized = False
        self._resources = {}

    async def initialize(self):
        """Initialize knowledge base resources"""
        if self._initialized:
            return

        try:
            await self._scan_knowledge_base()
            self._initialized = True
            logger.info(f"Knowledge base resource initialized with {len(self._resources)} files")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base resource: {str(e)}")
            raise

    async def list_resources(self) -> List[Resource]:
        """List all available knowledge base resources"""
        await self.initialize()

        resources = []

        for file_path, metadata in self._resources.items():
            resource = Resource(
                uri=f"knowledge://{metadata['category']}/{metadata['filename']}",
                name=metadata['display_name'],
                description=metadata['description'],
                mimeType="text/plain"
            )
            resources.append(resource)

        return resources

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read content of a specific knowledge base resource"""
        await self.initialize()

        try:
            # Parse URI: knowledge://category/filename
            if not uri.startswith("knowledge://"):
                raise ValueError(f"Invalid knowledge base URI: {uri}")

            path_part = uri[12:]  # Remove "knowledge://" prefix

            # Find matching resource
            matching_resource = None
            for file_path, metadata in self._resources.items():
                resource_path = f"{metadata['category']}/{metadata['filename']}"
                if resource_path == path_part:
                    matching_resource = (file_path, metadata)
                    break

            if not matching_resource:
                raise ValueError(f"Resource not found: {uri}")

            file_path, metadata = matching_resource

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text=content
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {str(e)}")
            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text=f"Error reading resource: {str(e)}"
                    )
                ]
            )

    async def _scan_knowledge_base(self):
        """Scan knowledge base directory and catalog resources"""
        knowledge_base_path = self.config.get_knowledge_base_path()

        if not knowledge_base_path.exists():
            logger.warning(f"Knowledge base directory not found: {knowledge_base_path}")
            return

        self._resources = {}

        # Scan for .txt files
        for txt_file in knowledge_base_path.glob("*.txt"):
            try:
                # Read file to get basic info
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determine category and create metadata
                category = self._determine_category(txt_file.name)
                display_name = self._create_display_name(txt_file.stem)
                description = self._create_description(txt_file.stem, content)

                self._resources[str(txt_file)] = {
                    'filename': txt_file.name,
                    'category': category,
                    'display_name': display_name,
                    'description': description,
                    'word_count': len(content.split()),
                    'char_count': len(content)
                }

            except Exception as e:
                logger.error(f"Error processing file {txt_file}: {str(e)}")
                continue

        logger.info(f"Scanned {len(self._resources)} knowledge base files")

    def _determine_category(self, filename: str) -> str:
        """Determine resource category from filename"""
        filename_lower = filename.lower()

        if any(term in filename_lower for term in ["company", "overview"]):
            return "company_info"
        elif any(term in filename_lower for term in ["performance", "past"]):
            return "past_performance"
        elif any(term in filename_lower for term in ["security", "compliance"]):
            return "security_compliance"
        elif any(term in filename_lower for term in ["technical", "product"]):
            return "technical_capabilities"
        elif any(term in filename_lower for term in ["management", "approach"]):
            return "management_approach"
        elif any(term in filename_lower for term in ["quality", "assurance"]):
            return "quality_assurance"
        elif any(term in filename_lower for term in ["personnel", "bio"]):
            return "personnel_bios"
        else:
            return "general"

    def _create_display_name(self, stem: str) -> str:
        """Create human-readable display name"""
        # Convert underscores to spaces and title case
        return stem.replace('_', ' ').replace('-', ' ').title()

    def _create_description(self, stem: str, content: str) -> str:
        """Create description for the resource"""
        # Get first sentence or first 100 characters as description
        sentences = content.split('. ')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 200:
                return first_sentence[:200] + "..."
            return first_sentence + "."

        # Fallback to truncated content
        return content[:100] + "..." if len(content) > 100 else content

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge base resources"""
        await self.initialize()

        if not self._resources:
            return {
                "total_files": 0,
                "total_words": 0,
                "categories": {},
                "largest_file": None
            }

        # Calculate statistics
        total_words = sum(meta['word_count'] for meta in self._resources.values())
        categories = {}
        largest_file = None
        largest_size = 0

        for file_path, metadata in self._resources.items():
            category = metadata['category']
            categories[category] = categories.get(category, 0) + 1

            if metadata['word_count'] > largest_size:
                largest_size = metadata['word_count']
                largest_file = metadata['display_name']

        return {
            "total_files": len(self._resources),
            "total_words": total_words,
            "average_words": total_words / len(self._resources),
            "categories": categories,
            "largest_file": largest_file,
            "largest_file_words": largest_size
        }