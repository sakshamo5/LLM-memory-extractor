import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AVAILABLE_MODELS = {
    "OpenAI GPT-OSS 20B": "openai/gpt-oss-20b",
    "OpenAI GPT OSS safegaurd 20b":"openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
}

DEFAULT_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.4
MAX_TOKENS = 2048

MIN_MESSAGES_FOR_EXTRACTION = 11  
MEMORY_EXTRACTION_TEMPERATURE = 0.1  

PERSONALITY_CONTEXT_WINDOW = 10  
