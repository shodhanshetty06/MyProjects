"""Centralized application configuration.

All environment-driven settings are declared here as a single Pydantic
Settings object. Nothing else in the codebase should read `os.environ`
directly — inject `get_settings()` instead so behavior stays testable
and overridable.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General ---
    ENVIRONMENT: Environment = Environment.LOCAL
    PROJECT_NAME: str = "AI Job Search Platform"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "0.1.0"

    # --- Security ---
    SECRET_KEY: str = "change-me-super-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ALGORITHM: str = "HS256"
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] | list[str] = Field(default_factory=list)

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # --- Database ---
    # Async driver (asyncpg) is used by the running application.
    DATABASE_URL: str = "postgresql+asyncpg://jobplatform:jobplatform@localhost:5432/jobplatform"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """Sync (psycopg) variant of DATABASE_URL, used by Alembic migrations."""
        return self.DATABASE_URL.replace("+asyncpg", "+psycopg")

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Celery ---
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # --- AWS S3 ---
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "job-platform-storage"
    S3_ENDPOINT_URL: str | None = None

    # --- Vector store ---
    # 1024 to match Voyage AI's `output_dimension` (see VOYAGE_MODEL below) —
    # not the OpenAI-sized 1536 placeholder from Phase 0.
    VECTOR_DIMENSIONS: int = 1024

    # --- Anthropic (Claude) ---
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-5"

    # --- Voyage AI (embeddings — Claude has no embeddings endpoint) ---
    VOYAGE_API_KEY: str | None = None
    VOYAGE_MODEL: str = "voyage-3.5"

    # --- Resume parser ---
    RESUME_MAX_UPLOAD_MB: int = 10

    # --- ATS Resume Generator ---
    # ats_score = keyword_score * KEYWORD_WEIGHT + structure_score * STRUCTURE_WEIGHT.
    ATS_KEYWORD_SCORE_WEIGHT: float = 0.6
    ATS_STRUCTURE_SCORE_WEIGHT: float = 0.4

    # --- Job Ingestion ---
    JOB_INGESTION_SWEEP_INTERVAL_MINUTES: int = 360
    JOB_INGESTION_STALE_CRAWL_HOURS: int = 24
    JOB_INGESTION_HTTP_TIMEOUT_SECONDS: int = 30
    JOB_INGESTION_MAX_JOBS_PER_RUN: int = 500
    JOB_INGESTION_USER_AGENT: str = "JobPlatformBot/1.0 (+https://jobplatform.example.com)"

    # --- Search ---
    SEARCH_CACHE_TTL_SECONDS: int = 90
    SEARCH_AUTOCOMPLETE_CACHE_TTL_SECONDS: int = 600
    SEARCH_TRENDING_CACHE_TTL_SECONDS: int = 1800
    SEARCH_AUTOCOMPLETE_MIN_CHARS: int = 2
    SEARCH_KEYWORD_CANDIDATE_LIMIT: int = 150
    SEARCH_SEMANTIC_CANDIDATE_LIMIT: int = 150
    SEARCH_HYBRID_KEYWORD_WEIGHT: float = 0.4
    SEARCH_HYBRID_SEMANTIC_WEIGHT: float = 0.6
    SEARCH_TRENDING_WINDOW_DAYS: int = 30
    SEARCH_TRGM_SIMILARITY_THRESHOLD: float = 0.3
    SEARCH_EXPERIENCE_LEVEL_HEURISTIC_ENABLED: bool = True

    # --- Matching ---
    # Base score = semantic (embedding cosine similarity) + skill-overlap,
    # weighted then scaled to 0..100; rule-based boosts are added on top.
    MATCHING_SEMANTIC_WEIGHT: float = 0.6
    MATCHING_SKILL_WEIGHT: float = 0.4
    # Individual rule-boost caps (points added to the 0..100 base score).
    MATCHING_BOOST_COUNTRY: float = 6.0
    MATCHING_BOOST_EXPERIENCE: float = 6.0
    MATCHING_BOOST_WORK_MODE: float = 4.0
    MATCHING_BOOST_SALARY: float = 6.0
    MATCHING_BOOST_EMPLOYMENT_TYPE: float = 3.0
    MATCHING_CANDIDATE_LIMIT: int = 100
    MATCHING_RECOMMENDATION_JOB_LIMIT: int = 20
    MATCHING_RECOMMENDATION_COMPANY_LIMIT: int = 10
    MATCHING_RECOMMENDATION_SKILL_LIMIT: int = 10
    MATCHING_RECOMMENDATION_REFRESH_INTERVAL_MINUTES: int = 1440
    MATCHING_STALE_MATCH_HOURS: int = 72

    # --- Auto Apply ---
    AUTO_APPLY_MAX_ATTEMPTS: int = 3
    AUTO_APPLY_BROWSER_TIMEOUT_MS: int = 30000
    AUTO_APPLY_CONFIRMATION_WAIT_MS: int = 2000
    AUTO_APPLY_WORKDAY_MAX_STEPS: int = 6

    # --- Gmail / Email Tracking ---
    GMAIL_OAUTH_CLIENT_ID: str | None = None
    GMAIL_OAUTH_CLIENT_SECRET: str | None = None
    GMAIL_OAUTH_REDIRECT_URI: str = "http://localhost:8000/api/v1/email-tracking/gmail/callback"
    GMAIL_OAUTH_SCOPES: str = "https://www.googleapis.com/auth/gmail.readonly"
    GMAIL_OAUTH_SUCCESS_REDIRECT_URL: str | None = None
    GMAIL_OAUTH_FAILURE_REDIRECT_URL: str | None = None
    GMAIL_OAUTH_STATE_TTL_SECONDS: int = 600
    GMAIL_TOKEN_ENCRYPTION_KEY: str | None = None
    GMAIL_SYNC_INTERVAL_MINUTES: int = 10
    GMAIL_MAX_MESSAGES_PER_SYNC: int = 50
    GMAIL_API_TIMEOUT_SECONDS: int = 20
    EMAIL_TRACKING_CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.6
    EMAIL_TRACKING_MATCH_CONFIDENCE_THRESHOLD: float = 0.72
    EMAIL_TRACKING_MATCH_MARGIN: float = 0.15
    EMAIL_TRACKING_MATCH_LOW_SCORE_FLOOR: float = 0.3
    EMAIL_TRACKING_MATCH_LOOKBACK_DAYS: int = 180
    EMAIL_TRACKING_COMPANY_MATCH_WEIGHT: float = 0.7
    EMAIL_TRACKING_TITLE_MATCH_WEIGHT: float = 0.3

    # --- AI Career Coach ---
    AI_CAREER_COACH_CHAT_MAX_HISTORY_MESSAGES: int = 20
    AI_CAREER_COACH_CHAT_MAX_TOKENS: int = 1500
    AI_CAREER_COACH_ANALYSIS_MAX_TOKENS: int = 4000
    AI_CAREER_COACH_NARRATIVE_MAX_TOKENS: int = 500
    AI_CAREER_COACH_MARKET_TREND_WINDOW_DAYS: int = 90
    AI_CAREER_COACH_SALARY_SAMPLE_LIMIT: int = 200
    AI_CAREER_COACH_SALARY_MIN_SAMPLE_SIZE: int = 3
    AI_CAREER_COACH_SKILL_GAP_REQUIRED_SKILL_LIMIT: int = 15
    AI_CAREER_COACH_COUNTRY_DEMAND_LIMIT: int = 15
    # job_readiness_score = resume_quality * RESUME_WEIGHT + skill_match * SKILL_WEIGHT
    #                       + completeness * COMPLETENESS_WEIGHT
    AI_CAREER_COACH_READINESS_RESUME_WEIGHT: float = 0.4
    AI_CAREER_COACH_READINESS_SKILL_WEIGHT: float = 0.35
    AI_CAREER_COACH_READINESS_COMPLETENESS_WEIGHT: float = 0.25
    AI_CAREER_COACH_INSIGHT_LIST_LIMIT: int = 20

    # --- Notification ---
    NOTIFICATION_SES_SENDER_EMAIL: str = "notifications@jobplatform.example.com"
    NOTIFICATION_SES_CONFIGURATION_SET: str | None = None
    NOTIFICATION_DEFAULT_CHANNELS: str = "in_app,email"
    NOTIFICATION_MAX_RETRIES: int = 3

    @property
    def notification_default_channels(self) -> list[str]:
        return [
            channel.strip()
            for channel in self.NOTIFICATION_DEFAULT_CHANNELS.split(",")
            if channel.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def is_local(self) -> bool:
        return self.ENVIRONMENT in (Environment.LOCAL, Environment.TEST)

    @property
    def auth_cookie_secure(self) -> bool:
        """Browsers silently drop `Secure` cookies over plain HTTP, so local
        Compose (no TLS) needs this off; anything else defaults to on."""
        return not self.is_local


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton for the process lifetime)."""
    return Settings()
