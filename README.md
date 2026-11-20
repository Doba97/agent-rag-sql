# Agent RAG et SQL : interroger documents et base de données en langage naturel

Un assistant qui répond à une question en anglais en choisissant le bon outil :

- **une recherche dans des documents** (RAG) : 3 015 passages Wikipedia ;
- **une requête SQL** sur une base de données : 19 bases SQLite du jeu Spider.

L'utilisateur pose sa question en langage naturel. Il n'a besoin ni de chercher dans les documents, ni d'écrire du SQL.

Chaque brique est **mesurée** sur des questions dont on connaît la bonne réponse, et chaque choix technique est justifié par ces mesures.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| 1. Cadrage | Question du projet, mesures, données | ✅ |
| 2. Mise en place | Structure, environnement, tests | ✅ |
| 3. Données | Téléchargement, nettoyage, séparation réglage / test | ✅ |
| 4. Recherche de passages | BM25, recherche dense, fusion RRF | ✅ |
| 5. Génération | Réponse de Qwen à partir des passages | ⏳ |
| 6. Outil SQL | Question → requête SQL, base en lecture seule | ⏳ |
| 7. Agent | Choix entre documents et SQL (routage) | ⏳ |
| 8. Analyse des erreurs | Traces Phoenix, classement des erreurs | ⏳ |
| 9. Déploiement | API FastAPI, interface Streamlit, Docker | ⏳ |

## Données

### Documents : rag-mini-wikipedia

- **3 200 passages** Wikipedia en anglais (pays, présidents américains, animaux, scientifiques…), déjà découpés : 48 mots en médiane.
- **185 passages de moins de 5 mots retirés** (légendes et tailles d'images, titres de section, comme `'125px'` ou `';Health rankings'`). Sur les questions de réglage, les scores sont identiques avec ou sans eux : on les retire car ils n'apportent aucune information. Il reste **3 015 passages**.
- Les passages étant déjà courts, aucun redécoupage n'a été nécessaire.
- Les accents mal encodés du jeu d'origine (`SÃ¶derby` au lieu de `Söderby`) sont réparés avec `ftfy`.

**Questions** : 918 au départ, **384 gardées**. On écarte :
- les réponses « yes » / « no », qu'on ne peut pas vérifier automatiquement (le mot apparaît partout) ;
- les réponses qui ne figurent dans aucun passage.

### Bases de données : Spider

- 1 034 questions de la partie *dev* de Spider, chacune avec sa requête SQL correcte.
- **19 bases SQLite**, de 2 à 11 tables (de `poker_player` à `student_transcripts_tracking`), et jusqu'à 531 377 lignes (`wta_1`).
- Chaque requête correcte est exécutée une fois au téléchargement pour vérifier les données.

### Séparation des questions

Tirage au hasard avec une graine fixe (42), donc reproductible :

| Jeu | Réglage | Test | Réserve |
|---|---|---|---|
| Documents | 30 | 100 | 254 |
| SQL | 30 | 100 | — |
| Routage | — | 200 (les deux jeux de test mélangés, chacun étiqueté « documents » ou « sql ») | — |

- **Réglage** : pour faire les choix et lire les erreurs.
- **Réserve** : pour confirmer les choix de la recherche de passages sur plus de questions (la recherche n'appelle pas le LLM, donc l'évaluation ne coûte rien).
- **Test** : utilisé une seule fois, à la fin. Les résultats ci-dessous n'en viennent pas.

## Étape 4 : recherche de passages

### Méthodes comparées

| Méthode | Principe |
|---|---|
| **BM25** | Mots-clés : favorise les passages qui contiennent les mots rares de la question |
| **BM25 + racinisation** | Idem, en réduisant les mots à leur racine (`died` → `die`) avec Snowball |
| **Recherche dense** | Sens : similarité cosinus entre vecteurs `BAAI/bge-base-en-v1.5` (768 dimensions, consigne bge ajoutée devant les questions) |
| **Combinée (RRF)** | Fusion des classements BM25 et dense (Reciprocal Rank Fusion, c = 60, 50 premiers résultats de chaque méthode) |

Avec 3 015 passages, les vecteurs sont stockés dans un tableau numpy : la recherche est exacte et instantanée. Ils sont calculés une seule fois puis mis en cache.

### Mesures

Un passage est considéré comme bon s'il contient la réponse attendue.

- **Hit@k** : part des questions dont la réponse figure dans les k premiers passages trouvés.
- **MRR@10** : 1 si le premier bon passage est 1er, 1/2 s'il est 2e, etc.

Precision@k et Recall@k ne sont pas calculés : ils supposent de connaître **tous** les bons passages de chaque question, et rag-mini-wikipedia ne fournit pas cette information. Hit@k est la mesure habituelle pour la recherche de passages en questions-réponses (*top-k accuracy* dans l'article DPR). Il est égal au Recall@k quand chaque question n'a qu'un seul bon passage.

### Résultats sur les 254 questions de la réserve

| Méthode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 0,65 | 0,79 | 0,85 | 0,88 | 0,73 |
| BM25 + racinisation | 0,67 | 0,81 | 0,86 | 0,89 | 0,75 |
| Recherche dense | 0,65 | 0,82 | 0,86 | 0,89 | 0,74 |
| Combinée (RRF) | 0,72 | 0,82 | 0,85 | 0,91 | 0,78 |
| **Combinée (RRF) + racinisation** | **0,72** | **0,83** | **0,86** | **0,92** | **0,79** |

**Méthode retenue : combinée (RRF) + racinisation.**

### Ce que montrent les résultats

1. **La combinaison place plus souvent le bon passage en tête** : 72 % contre 65 à 67 % pour chaque méthode seule (13 à 18 questions de plus sur 254). Les deux méthodes ratent des questions différentes : BM25 est fort sur les noms propres et les dates, la recherche dense sur les reformulations.
2. **La racinisation apporte environ 1 point**, toujours dans le même sens, et corrige un défaut identifié en lisant les erreurs : sans elle, BM25 ne relie pas « die » dans la question à « died » dans le passage. Elle ne relie pas pour autant « presidency » et « presidential ».
3. **À 5 passages, les méthodes se valent** (environ 86 %) : les différences se jouent dans les premières positions.
4. **30 questions ne suffisent pas pour choisir.** Sur les questions de réglage, la méthode combinée atteignait 97 % dans le top 3, contre 83 % sur la réserve. D'où l'usage de la réserve pour confirmer les choix.

### Limites connues

- **La mesure peut se tromper sur les réponses très courtes.** Pour « How many municipalities are within Oberland? » (réponse : 6), le premier passage trouvé dit « 11 municipalities… five of them fall within Unterland, and the remainder within Oberland » : il contient la réponse, mais pas le chiffre 6. Les questions ratées du jeu de test seront relues à la main.
- **Certains passages ont perdu leur sujet.** « The life expectancy is 82 years for women and 75 years for men » ne dit pas de quel pays il s'agit. Ajouter le titre de l'article devant chaque passage corrigerait ce défaut, mais le jeu de données ne fournit pas les titres.
- **Environ 8 % des questions** n'ont aucun bon passage dans les 10 premiers résultats, en partie à cause de la limite de mesure ci-dessus.

## Installation et reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # puis mettre sa clé Together dans .env (utile à partir de l'étape 5)

pytest                                            # 37 tests, sans réseau ni clé
python -m src.data                                # télécharge et sépare les données
python -m src.eval_retrieval                      # recherche : questions de réglage
python -m src.eval_retrieval --questions reserve  # recherche : questions de la réserve
```

Le premier lancement de `eval_retrieval` calcule les vecteurs des passages, ce qui peut être long sur un ordinateur sans carte graphique. Les lancements suivants les relisent depuis `data/passage_vectors.npz`.

## Structure

```
src/
  config.py           paramètres (modèles, dossiers)
  data.py             téléchargement, nettoyage, séparation réglage / test / réserve
  embeddings.py       vecteurs avec BAAI/bge-base-en-v1.5
  retrieval.py        BM25, recherche dense, fusion RRF
  evaluation.py       Hit@k, MRR
  eval_retrieval.py   comparaison des méthodes de recherche
  llm.py              appel de Qwen par l'API Together
tests/                tests pytest (sans réseau)
notebooks/            exploration des données
results/              tableaux de résultats
data/                 données téléchargées (non versionnées, recréées par src.data)
```

## Outils

Python · rank_bm25 · snowballstemmer · sentence-transformers · numpy · Hugging Face datasets · Together API (Qwen) · pytest

## Références

- Yu et al. (2018), *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*.
- Cormack, Clarke et Büttcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*.
- Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering*.
- Jeu de données [rag-datasets/rag-mini-wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) ; modèle [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5).
