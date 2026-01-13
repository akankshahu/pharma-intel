# Configuration module for Pharma-Intellect
# Loads environment variables and provides configuration constants

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# OpenAI Configuration
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ============================================
# API Configuration
# ============================================
PUBMED_MAX_RESULTS = int(os.getenv("PUBMED_MAX_RESULTS", "100"))
PUBMED_KEYWORDS = os.getenv(
    "PUBMED_KEYWORDS",
    "oncology drug discovery,autoimmune disorders,risperidone,drug repurposing"
).split(",")

CLINICAL_TRIALS_MAX_RESULTS = int(os.getenv("CLINICAL_TRIALS_MAX_RESULTS", "100"))
CLINICAL_TRIALS_CONDITIONS = os.getenv(
    "CLINICAL_TRIALS_CONDITIONS",
    "Oncology,Autoimmune Disorders,Hypertension"
).split(",")

# ============================================
# Vector Database Configuration
# ============================================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "pharma_intellect_knowledge_base")

# ============================================
# Embedding Model Configuration
# ============================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# This model produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384

# ============================================
# RAG Configuration
# ============================================
NUM_SOURCES_TO_RETRIEVE = int(os.getenv("NUM_SOURCES_TO_RETRIEVE", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ============================================
# Streamlit Configuration
# ============================================
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_HEADLESS = os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() == "true"

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ============================================
# API Endpoints
# ============================================
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CLINICAL_TRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# ============================================
# Application Constants
# ============================================
APP_NAME = "Pharma-Intellect"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered Drug Discovery Research Assistant"

def validate_config():
    """Validate critical configuration values."""
    errors = []

    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-proj-your-api-key-here":
        errors.append("OPENAI_API_KEY is not set. Please configure it in .env file")

    if LLM_MODEL not in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:
        errors.append(f"Invalid LLM_MODEL: {LLM_MODEL}")

    return errors

if __name__ == "__main__":
    # Test configuration
    errors = validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("✅ Configuration is valid!")
        print(f"   - OpenAI Model: {LLM_MODEL}")
        print(f"   - Embedding Model: {EMBEDDING_MODEL}")
        print(f"   - PubMed Keywords: {len(PUBMED_KEYWORDS)} keywords")
        print(f"   - Clinical Trials Conditions: {len(CLINICAL_TRIALS_CONDITIONS)} conditions")
