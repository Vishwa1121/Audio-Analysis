import os
import logging
import warnings
import openai

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# OpenAI API key loading
openai_api_key = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        logger.info("OpenAI API key loaded successfully from .env file")
    else:
        logger.warning("OPENAI_API_KEY not found in .env file")
except UnicodeDecodeError:
    logger.error("Error: .env file is corrupted or not in UTF-8 format")
    logger.error("Please check your .env file or create a new one")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")

# Fallback to environment variables
if not openai_api_key:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        logger.info("OpenAI API key loaded from environment variables")
    else:
        logger.warning("OpenAI API key not found in environment variables")


def get_cors_origins() -> list[str]:
    """Return list of allowed CORS origins (extend via env if needed)."""
    origins = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ]
    extra = os.getenv("CORS_EXTRA_ORIGINS", "").strip()
    if extra:
        origins.extend([o for o in (x.strip() for x in extra.split(",")) if o])
    return origins


