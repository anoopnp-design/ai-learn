from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Enterprise AI Serving Architecture"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    # Model Configuration
    MODEL_NAME: str = "iris_classifier"
    MODEL_VERSION: str = "v1"
    
    # Enable automatic configuration binding from a .env file if it exists
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings();