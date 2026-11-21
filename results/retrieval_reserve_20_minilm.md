Recherche de passages : 20 questions de la réserve, 3015 passages (passages d'au moins 5 mots), modèle d'embeddings BAAI/bge-base-en-v1.5

| Méthode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (mots-clés) | 0.60 | 0.75 | 0.80 | 0.85 | 0.69 |
| BM25 + racinisation | 0.55 | 0.80 | 0.80 | 0.80 | 0.67 |
| Embeddings (sens) | 0.60 | 0.60 | 0.65 | 0.70 | 0.62 |
| Combinée (RRF) | 0.65 | 0.80 | 0.80 | 0.90 | 0.73 |
| Combinée (RRF) + racinisation | 0.70 | 0.80 | 0.85 | 0.90 | 0.76 |
| Combinée + racinisation + reranking minilm (10 candidats) | 0.75 | 0.85 | 0.90 | 0.90 | 0.81 |
