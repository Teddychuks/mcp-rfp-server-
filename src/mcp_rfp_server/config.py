"""
Configuration management for MCP-RFP Server with SharePoint Integration
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Unified server configuration with SharePoint support"""

    # Server identity
    server_name: str = Field(default="mcp-rfp-server", env="SERVER_NAME")
    server_version: str = Field(default="1.0.0", env="SERVER_VERSION")

    # Gemini API configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash-latest", env="GEMINI_MODEL")

    # Knowledge base source configuration
    knowledge_source: str = Field(default="local", env="KNOWLEDGE_SOURCE")  # "local" or "sharepoint"

    # Local paths configuration (used when knowledge_source="local")
    knowledge_base_path: str = Field(default="./knowledge_base", env="KNOWLEDGE_BASE_PATH")

    # SharePoint configuration (used when knowledge_source="sharepoint")
    sharepoint_tenant_id: Optional[str] = Field(default=None, env="SHAREPOINT_TENANT_ID")
    sharepoint_client_id: Optional[str] = Field(default=None, env="SHAREPOINT_CLIENT_ID")
    sharepoint_client_secret: Optional[str] = Field(default=None, env="SHAREPOINT_CLIENT_SECRET")
    sharepoint_site_url: Optional[str] = Field(default=None, env="SHAREPOINT_SITE_URL")
    sharepoint_folder_path: str = Field(default="knowledge_base", env="SHAREPOINT_FOLDER_PATH")

    # Other paths
    chroma_db_path: str = Field(default="./chroma_data", env="CHROMA_DB_PATH")
    output_path: str = Field(default="./outputs", env="OUTPUT_PATH")

    # Processing limits
    max_document_size_mb: int = Field(default=50, env="MAX_DOCUMENT_SIZE_MB")
    max_requirements_per_document: int = Field(default=200, env="MAX_REQUIREMENTS_PER_DOCUMENT")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    max_search_results: int = Field(default=50, env="MAX_SEARCH_RESULTS")

    # NLP and ML settings
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    # Search and processing parameters
    default_search_limit: int = Field(default=5, env="DEFAULT_SEARCH_LIMIT")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    confidence_threshold: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD")

    # Logging and debug
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    # Knowledge base categories
    knowledge_categories: List[str] = Field(
        default=[
            "company_overview",
            "past_performance",
            "technical_capabilities",
            "security_compliance",
            "management_approach",
            "quality_assurance",
            "personnel_bios",
            "product_info"
        ]
    )

    # MCP-specific settings
    mcp_protocol_version: str = Field(default="2024-11-05", env="MCP_PROTOCOL_VERSION")
    enable_mcp_logging: bool = Field(default=True, env="ENABLE_MCP_LOGGING")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables"""
        return cls()

    def get_knowledge_base_path(self) -> Path:
        """Get knowledge base path as Path object (only for local source)"""
        return Path(self.knowledge_base_path)

    def get_chroma_db_path(self) -> Path:
        """Get ChromaDB path as Path object"""
        return Path(self.chroma_db_path)

    def get_output_path(self) -> Path:
        """Get output path as Path object"""
        return Path(self.output_path)

    def is_sharepoint_enabled(self) -> bool:
        """Check if SharePoint is enabled and properly configured"""
        return (
            self.knowledge_source.lower() == "sharepoint" and
            all([
                self.sharepoint_tenant_id,
                self.sharepoint_client_id,
                self.sharepoint_client_secret,
                self.sharepoint_site_url
            ])
        )

    def get_sharepoint_config(self) -> dict:
        """Get SharePoint configuration as dictionary"""
        if not self.is_sharepoint_enabled():
            raise ValueError("SharePoint is not properly configured")

        return {
            "tenant_id": self.sharepoint_tenant_id,
            "client_id": self.sharepoint_client_id,
            "client_secret": self.sharepoint_client_secret,
            "site_url": self.sharepoint_site_url,
            "folder_path": self.sharepoint_folder_path
        }

    def ensure_directories(self):
        """Ensure all required directories exist (for local paths only)"""
        directories = [
            self.get_chroma_db_path(),
            self.get_output_path()
        ]

        # Only create knowledge_base directory if using local source
        if self.knowledge_source.lower() == "local":
            directories.append(self.get_knowledge_base_path())

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_configuration(self) -> bool:
        """Validate configuration settings"""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        if self.max_document_size_mb <= 0:
            raise ValueError("MAX_DOCUMENT_SIZE_MB must be positive")

        # Validate knowledge source configuration
        if self.knowledge_source.lower() == "sharepoint":
            if not self.is_sharepoint_enabled():
                missing_vars = []
                if not self.sharepoint_tenant_id:
                    missing_vars.append("SHAREPOINT_TENANT_ID")
                if not self.sharepoint_client_id:
                    missing_vars.append("SHAREPOINT_CLIENT_ID")
                if not self.sharepoint_client_secret:
                    missing_vars.append("SHAREPOINT_CLIENT_SECRET")
                if not self.sharepoint_site_url:
                    missing_vars.append("SHAREPOINT_SITE_URL")

                raise ValueError(f"SharePoint configuration incomplete. Missing: {', '.join(missing_vars)}")

        elif self.knowledge_source.lower() == "local":
            # Check that local knowledge base path exists
            if not self.get_knowledge_base_path().exists():
                raise ValueError(f"Local knowledge base directory does not exist: {self.knowledge_base_path}")

        else:
            raise ValueError(f"Invalid knowledge_source: {self.knowledge_source}. Must be 'local' or 'sharepoint'")

        # Check that other critical paths exist
        for path_name, path_value in [("chroma_db_path", self.chroma_db_path), ("output_path", self.output_path)]:
            if not Path(path_value).exists():
                raise ValueError(f"Required path does not exist: {path_name} = {path_value}")

        return True