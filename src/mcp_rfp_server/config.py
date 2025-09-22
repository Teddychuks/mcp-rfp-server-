"""
Configuration management for MCP-RFP Server
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Unified server configuration"""

    # Server identity
    server_name: str = Field(default="mcp-rfp-server", env="SERVER_NAME")
    server_version: str = Field(default="1.0.0", env="SERVER_VERSION")

    # Gemini API configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash-latest", env="GEMINI_MODEL")

    # Paths configuration
    knowledge_base_path: str = Field(default="./knowledge_base", env="KNOWLEDGE_BASE_PATH")
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
        """Get knowledge base path as Path object"""
        return Path(self.knowledge_base_path)

    def get_chroma_db_path(self) -> Path:
        """Get ChromaDB path as Path object"""
        return Path(self.chroma_db_path)

    def get_output_path(self) -> Path:
        """Get output path as Path object"""
        return Path(self.output_path)

    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.get_knowledge_base_path(),
            self.get_chroma_db_path(),
            self.get_output_path()
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_configuration(self) -> bool:
        """Validate configuration settings"""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        if self.max_document_size_mb <= 0:
            raise ValueError("MAX_DOCUMENT_SIZE_MB must be positive")

        # Check that all critical paths exist
        for path_name in ["knowledge_base_path", "chroma_db_path", "output_path"]:
            path_value = getattr(self, path_name)
            if not Path(path_value).exists():
                raise ValueError(f"Required path does not exist: {path_name} = {path_value}")

        return True