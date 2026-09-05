from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "ai-evaluation-platform"
    GEMINI_API_KEY: str = ""
    # Add other configuration variables here as we go
    
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings = Settings()
