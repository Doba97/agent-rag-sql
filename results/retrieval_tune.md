Recherche de passages : 30 questions de réglage, 3015 passages (passages d'au moins 5 mots), modèle d'embeddings BAAI/bge-base-en-v1.5

| Méthode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (mots-clés) | 0.70 | 0.83 | 0.87 | 0.97 | 0.78 |
| BM25 + racinisation | 0.73 | 0.87 | 0.93 | 0.97 | 0.81 |
| Embeddings (sens) | 0.60 | 0.87 | 0.90 | 0.97 | 0.74 |
| Combinée (RRF) | 0.70 | 0.97 | 0.97 | 0.97 | 0.82 |
| Combinée (RRF) + racinisation | 0.73 | 0.93 | 0.97 | 0.97 | 0.83 |
