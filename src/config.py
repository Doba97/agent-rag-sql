"""Paramètres du projet, lus dans les variables d'environnement (fichier .env)."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv est facultatif
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / ".models"

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
