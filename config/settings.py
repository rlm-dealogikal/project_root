# /app/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE_PATH, "data/raw/csv")

JSON_PATH = os.path.join(BASE_PATH, "data/raw/json")

CHROMA_PATH = os.path.join(BASE_PATH, "vector_db/chroma")

COLLECTION_NAME = "spreadsheet_collection"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

USER_INPUT_API_URL = os.getenv("USER_INPUT_API_URL")

MISTRAL_MODEL_ID = "openai/gpt-3.5-turbo"