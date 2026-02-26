"""
Configuration Management for Production RAG System

Loads configuration from environment variables with validation.
Uses pydantic-settings for type safety and validation.

Usage:
    from config import settings
    
    print(settings.OPENAI_API_KEY)
    print(settings.CHUNK_SIZE)
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ─── LLM & Embeddings ────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key (required)")
    OPENAI_ORG_ID: Optional[str] = Field(None, description="OpenAI organization ID")
    
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    LLM_MODEL: str = Field(
        default="gpt-4o-mini",
        description="LLM model for generation"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0=deterministic, 2=creative)"
    )
    LLM_MAX_TOKENS: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="Max tokens in LLM response"
    )
    
    # ─── Vector Database ─────────────────────────────────────────────────────
    PGVECTOR_HOST: str = Field(default="localhost")
    PGVECTOR_PORT: int = Field(default=5432, ge=1, le=65535)
    PGVECTOR_DB: str = Field(default="rag_system")
    PGVECTOR_USER: str = Field(default="rag_user")
    PGVECTOR_PASSWORD: str = Field(..., description="PostgreSQL password (required)")
    PGVECTOR_SCHEMA: str = Field(default="public")
    
    PGVECTOR_POOL_SIZE: int = Field(default=5, ge=1)
    PGVECTOR_MAX_OVERFLOW: int = Field(default=10, ge=0)
    
    @property
    def PGVECTOR_CONNECTION_STRING(self) -> str:
        """Build PostgreSQL connection string"""
        return (
            f"postgresql://{self.PGVECTOR_USER}:{self.PGVECTOR_PASSWORD}"
            f"@{self.PGVECTOR_HOST}:{self.PGVECTOR_PORT}/{self.PGVECTOR_DB}"
        )
    
    # ─── Retrieval Configuration ─────────────────────────────────────────────
    RETRIEVAL_VECTOR_TOP_K: int = Field(default=50, ge=1, le=100)
    VECTOR_SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    
    RETRIEVAL_BM25_TOP_K: int = Field(default=50, ge=1, le=100)
    
    RERANK_TOP_K: int = Field(default=5, ge=1, le=20)
    RERANK_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    RRF_K: int = Field(default=60, ge=1, description="RRF constant")
    
    # ─── Chunking ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=512, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=51, ge=0)
    CHUNKING_STRATEGY: str = Field(
        default="sentence_based",
        pattern="^(sentence_based|fixed_size|semantic)$"
    )
    
    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size"""
        chunk_size = info.data.get("CHUNK_SIZE", 512)
        if v >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP ({v}) must be < CHUNK_SIZE ({chunk_size})")
        return v
    
    # ─── Evaluation ──────────────────────────────────────────────────────────
    RAGAS_FAITHFULNESS_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)
    RAGAS_ANSWER_RELEVANCY_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)
    RAGAS_CONTEXT_RELEVANCY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    
    EVAL_SAMPLE_RATE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of queries to evaluate"
    )
    EVAL_DATASET_PATH: str = Field(default="data/eval/golden_questions.json")
    
    # ─── Monitoring ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    LOG_FORMAT: str = Field(
        default="json",
        pattern="^(json|text)$"
    )
    
    # Tracing (optional)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com")
    ENABLE_TRACING: bool = Field(default=False)
    
    TRACK_COSTS: bool = Field(default=True)
    ALERT_DAILY_COST_THRESHOLD: float = Field(default=100.0, ge=0.0)
    
    # ─── Guardrails ──────────────────────────────────────────────────────────
    ENABLE_PII_DETECTION: bool = Field(default=True)
    ENABLE_PROMPT_INJECTION_CHECK: bool = Field(default=True)
    MAX_QUERY_LENGTH: int = Field(default=2000, ge=1, le=10000)
    
    # ─── API Server ──────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    API_WORKERS: int = Field(default=4, ge=1)
    API_RELOAD: bool = Field(default=False)
    
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"])
    
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1)
    RATE_LIMIT_PERIOD: int = Field(default=60, ge=1)
    
    # ─── Storage ─────────────────────────────────────────────────────────────
    SOURCE_DOCS_PATH: str = Field(default="data/sample_docs")
    LOGS_PATH: str = Field(default="logs")
    EVAL_RESULTS_PATH: str = Field(default="outputs/eval_results")
    
    # ─── Environment ─────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(
        default="development",
        pattern="^(development|staging|production)$"
    )
    DEBUG: bool = Field(default=False)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"


# ─── Singleton Instance ──────────────────────────────────────────────────────
settings = Settings()


# ─── Validation on Import ────────────────────────────────────────────────────
def validate_configuration():
    """Validate configuration on import"""
    try:
        _ = settings.OPENAI_API_KEY
        _ = settings.PGVECTOR_PASSWORD
    except Exception as e:
        print(f"⚠️  Configuration error: {e}")
        print("Make sure .env file exists and contains required values")
        print("Copy .env.example to .env and fill in your credentials")
        raise


# Run validation
validate_configuration()
