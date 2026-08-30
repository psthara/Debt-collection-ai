from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RAG_DIR = ROOT / "rag" / "index"
DOC_DIR = ROOT / "rag" / "documents"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
