from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    GEMINI_API_KEY: str
    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "storyboard_generator"
    STORAGE_DIR: str = "storage"
    VITE_DEV_ORIGIN: str = "http://localhost:5173"

    # Gemini model identifiers
    GEMINI_TEXT_MODEL: str = "gemini-3.5-flash"
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image"

    # Image processing
    IMAGE_MAX_LONG_EDGE: int = 512

    # Context cache TTL in seconds (1 hour default)
    CACHE_TTL_SECONDS: int = 3600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
