import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

CONFIG = load_config()

# Secrets from environment
class Settings:
    WHATSAPP_ACCESS_TOKEN: str = os.getenv("WHATSAPP_ACCESS_TOKEN")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN")
    WHATSAPP_WABA_ID: str = os.getenv("WHATSAPP_WABA_ID")
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    NARAD_API_KEY: str = os.getenv("NARAD_API_KEY", "")
    PUBLIC_SERVER_URL: str = os.getenv("PUBLIC_SERVER_URL", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")


    

settings = Settings()