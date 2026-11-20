Recherche de passages : 254 questions de la réserve, 3015 passages (passages d'au moins 5 mots), modèle d'embeddings BAAI/bge-base-en-v1.5

| Méthode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (mots-clés) | 0.65 | 0.79 | 0.85 | 0.88 | 0.73 |
| BM25 + racinisation | 0.67 | 0.81 | 0.86 | 0.89 | 0.75 |
| Embeddings (sens) | 0.65 | 0.82 | 0.86 | 0.89 | 0.74 |
| Combinée (RRF) | 0.72 | 0.82 | 0.85 | 0.91 | 0.78 |
| Combinée (RRF) + racinisation | 0.72 | 0.83 | 0.86 | 0.92 | 0.79 |
