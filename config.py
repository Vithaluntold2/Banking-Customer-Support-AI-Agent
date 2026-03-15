import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    """Check Streamlit secrets first (for Cloud), then fall back to env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# azure openai creds
AZURE_OPENAI_ENDPOINT = _get("AZURE_OPENAI_ENDPOINTS")
AZURE_OPENAI_API_KEY = _get("AZURE_OPENAI_API_KEYS")
AZURE_OPENAI_DEPLOYMENT = _get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
AZURE_OPENAI_API_VERSION = _get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# paths
BASE_DIR = os.path.dirname(__file__)
DATABASE_PATH = os.path.join(BASE_DIR, "data", "support_tickets.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "agent_logs.json")
