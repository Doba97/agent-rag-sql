# Agent RAG et SQL : interroger documents et base de données en langage naturel

Projet en cours. Un agent répond à une question en choisissant entre deux outils :
une recherche dans des documents (RAG) et une requête SQL sur une base de données.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # puis mettre sa clé Together dans .env
pytest
```

## Structure

```
src/
  config.py       paramètres (modèles, dossiers)
  llm.py          appel de Qwen par l'API Together
  embeddings.py   embeddings avec BAAI/bge-base-en-v1.5
tests/            tests pytest (sans réseau)
data/             données téléchargées (non versionnées)
notebooks/        explorations
```
