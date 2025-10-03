"""
Configuration management for MCP-RFP Server with SharePoint & Google Drive Integration
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Unified server configuration with SharePoint and Google Drive support"""

    # Server identity
    server_name: str = Field(default="mcp-rfp-server", env="SERVER_NAME")
    server_version: str = Field(default="1.0.0", env="SERVER_VERSION")

    # Gemini API configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash-latest", env="GEMINI_MODEL")

    # Knowledge base source configuration
    knowledge_source: str = Field(default="local", env="KNOWLEDGE_SOURCE")  # "local", "sharepoint", or "gdrive"

    # Local paths configuration (used when knowledge_source="local")
    knowledge_base_path: str = Field(default="./knowledge_base", env="KNOWLEDGE_BASE_PATH")

    # SharePoint configuration (used when knowledge_source="sharepoint")
    sharepoint_tenant_id: Optional[str] = Field(default=None, env="SHAREPOINT_TENANT_ID")
    sharepoint_client_id: Optional[str] = Field(default=None, env="SHAREPOINT_CLIENT_ID")
    sharepoint_client_secret: Optional[str] = Field(default=None, env="SHAREPOINT_CLIENT_SECRET")
    sharepoint_site_url: Optional[str] = Field(default=None, env="SHAREPOINT_SITE_URL")
    sharepoint_folder_path: str = Field(default="knowledge_base", env="SHAREPOINT_FOLDER_PATH")

    # Google Drive configuration (used when knowledge_source="gdrive")
    gdrive_knowledge_base_folder_id: Optional[str] = Field(default=None, env="GDRIVE_KNOWLEDGE_BASE_FOLDER_ID")

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
            "company_overview", "past_performance", "technical_capabilities",
            "security_compliance", "management_approach", "quality_assurance",
            "personnel_bios", "product_info"
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
        """Get knowledge base path as Path object"""
        return Path(self.knowledge_base_path)

    def get_chroma_db_path(self) -> Path:
        """Get ChromaDB path as Path object"""
        return Path(self.chroma_db_path)

    def get_output_path(self) -> Path:
        """Get output path as Path object"""
        return Path(self.output_path)

    def is_sharepoint_enabled(self) -> bool:
        """Check if SharePoint is the configured source."""
        return self.knowledge_source.lower() == "sharepoint"

    def is_gdrive_enabled(self) -> bool:
        """Check if Google Drive is the configured source."""
        return self.knowledge_source.lower() == "gdrive"

    def get_sharepoint_config(self) -> dict:
        """Get SharePoint configuration as a dictionary."""
        if not self.is_sharepoint_enabled():
            raise ValueError("SharePoint source is not enabled.")
        return {
            "tenant_id": self.sharepoint_tenant_id,
            "client_id": self.sharepoint_client_id,
            "client_secret": self.sharepoint_client_secret,
            "site_url": self.sharepoint_site_url,
            "folder_path": self.sharepoint_folder_path,
        }

    def get_gdrive_config(self) -> dict:
        """Get Google Drive configuration as a dictionary."""
        if not self.is_gdrive_enabled():
            raise ValueError("Google Drive source is not enabled.")
        return {
            "folder_id": self.gdrive_knowledge_base_folder_id
        }

    def ensure_directories(self):
        """Ensure all required local directories exist."""
        directories = [
            self.get_chroma_db_path(),
            self.get_output_path()
        ]
        if self.knowledge_source.lower() == "local":
            directories.append(self.get_knowledge_base_path())

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @model_validator(mode='after')
    def validate_configuration(self) -> 'ServerConfig':
        """Validate the entire configuration after loading."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is a required environment variable.")

        source = self.knowledge_source.lower()

        if source == "sharepoint":
            sp_vars = [
                self.sharepoint_tenant_id, self.sharepoint_client_id,
                self.sharepoint_client_secret, self.sharepoint_site_url
            ]
            if not all(sp_vars):
                raise ValueError("For SharePoint source, all SHAREPOINT_* variables are required.")

        elif source == "gdrive":
            if not self.gdrive_knowledge_base_folder_id:
                raise ValueError("For Google Drive source, GDRIVE_KNOWLEDGE_BASE_FOLDER_ID is required.")

        elif source == "local":
            pass

        else:
            raise ValueError(f"Invalid KNOWLEDGE_SOURCE: '{self.knowledge_source}'. Must be 'local', 'sharepoint', or 'gdrive'.")

        return self