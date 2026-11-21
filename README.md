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
| 3. Données | Téléchargement, nettoyage, séparation réglage / test / réserve | ✅ |
| 4. Recherche de passages | BM25, recherche dense, fusion RRF, reranking | ✅ |
| 5. Génération | Réponse du LLM à partir des passages, juge LLM | ✅ |
| 6. Outil SQL | Question → requête SQL, base en lecture seule | ⏳ |
| 7. Agent | Choix entre documents et SQL (routage) | ⏳ |
| 8. Analyse des erreurs | Traces Phoenix, classement des erreurs | ⏳ |
| 9. Déploiement | API FastAPI, interface Streamlit, Docker | ⏳ |

## Chaîne retenue pour les documents

```
Question
  → recherche combinée : BM25 (avec racinisation) + recherche dense (bge-base-en-v1.5), fusion RRF
  → 10 passages candidats
  → reranking : cross-encoder ms-marco-MiniLM-L-6-v2
  → 3 meilleurs passages, numérotés, envoyés au LLM avec la question
  → réponse courte avec ses sources ([1], [2]...), ou « NOT FOUND »
```

Résultat sur 100 questions hors test : **89 % de réponses justes** (selon un juge LLM), avec environ 500 tokens par question.

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
- **Réserve** : pour confirmer les choix sur plus de questions. 30 questions se sont révélées trop peu nombreuses : elles donnaient 97 % dans le top 3 pour la recherche combinée, contre 83 % sur la réserve.
- **Test** : utilisé une seule fois, à la fin. **Aucun résultat ci-dessous n'en vient.**

## Étape 4 : recherche de passages

### Méthodes comparées

| Méthode | Principe |
|---|---|
| **BM25** | Mots-clés : favorise les passages qui contiennent les mots rares de la question |
| **BM25 + racinisation** | Idem, en réduisant les mots à leur racine (`died` → `die`) avec Snowball |
| **Recherche dense** | Sens : similarité cosinus entre vecteurs `BAAI/bge-base-en-v1.5` (768 dimensions, consigne bge ajoutée devant les questions) |
| **Combinée (RRF)** | Fusion des classements BM25 et dense (Reciprocal Rank Fusion, c = 60, 50 premiers résultats de chaque méthode) |
| **Reranking** | Un cross-encoder relit les 10 premiers passages de la recherche combinée **avec** la question et les renote |

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
| Combinée (RRF) + racinisation | 0,72 | 0,83 | 0,86 | 0,92 | 0,79 |
| **+ reranking MiniLM (10 candidats)** | **0,82** | **0,89** | **0,90** | **0,92** | **0,86** |
| + reranking bge-reranker-base (10 candidats) | 0,62 | 0,83 | 0,87 | 0,92 | 0,73 |

### Ce que montrent les résultats

1. **La combinaison place plus souvent le bon passage en tête** : 72 % contre 65 à 67 % pour chaque méthode seule. Les deux méthodes ratent des questions différentes : BM25 est fort sur les noms propres et les dates, la recherche dense sur les reformulations.
2. **La racinisation apporte environ 1 point**, toujours dans le même sens, et corrige un défaut identifié en lisant les erreurs : sans elle, BM25 ne relie pas « die » dans la question à « died » dans le passage. Elle ne relie pas pour autant « presidency » et « presidential ».
3. **Le reranking MiniLM est le plus grand gain** : bon passage en tête pour 82 % des questions au lieu de 72 % (environ 25 questions de plus), et dans le top 3 pour 89 % au lieu de 83 %. Il ne fait que réordonner les 10 candidats, donc il ne peut pas dépasser Hit@10 (92 %) : on est proche de ce plafond.
4. **Un reranker plus gros n'est pas forcément meilleur** : bge-reranker-base fait moins bien que l'absence de reranking en 1re position. Une explication possible, non vérifiée : MiniLM a été entraîné sur MS MARCO (questions réelles et passages qui y répondent), exactement notre tâche.

## Étape 5 : génération des réponses

### Principe

Les 3 passages retenus sont numérotés et envoyés au LLM avec la question. Les consignes lui demandent de répondre en quelques mots, **uniquement à partir des passages**, de citer ses sources, et de répondre « NOT FOUND » si la réponse n'y est pas.

Modèle : `gpt-6-luna` (OpenAI), température 0. Le code fonctionne avec toute plateforme compatible avec l'API d'OpenAI (OpenRouter, Together, Groq…) : il suffit de changer le fichier `.env`.

### Mesures

| Mesure | Ce qu'elle dit |
|---|---|
| Réponse juste (mot à mot) | Les mots de la réponse attendue figurent dans la réponse donnée (articles ignorés) |
| **Juste selon le juge** | Un LLM compare la réponse à la référence et décide si elle donne le même fait |
| F1 mots | Note partielle sur les mots communs (mesure du jeu SQuAD) |
| Bon passage fourni | La réponse attendue figure dans un des passages envoyés |
| Refus avec bon passage | Le modèle répond « NOT FOUND » alors que la réponse était dans les passages |

**Pourquoi un juge ?** Certaines réponses de référence sont des phrases entières (« While as of October 2007, Canada's national unemployment rate of 5.9%… ») : la réponse « 5.9% » est juste, mais une comparaison mot à mot la compte fausse. Sur les questions de réglage, 18 réponses sur 30 étaient justes mot à mot, contre 26 selon le juge.

**Le juge a été relu à la main** sur les 30 questions de réglage : ses verdicts sont cohérents, sauf pour 2 questions dont la référence est une longue phrase mal alignée avec la question, où il donne des verdicts différents pour des réponses presque identiques. Marge d'incertitude : environ 1 à 2 questions sur 30.

### Résultats sur 100 questions de la réserve

| Configuration | Juste selon le juge | Mot à mot | F1 mots | Bon passage fourni | Tokens par question |
|---|---|---|---|---|---|
| 5 passages, sans reranking | 0,85 | 0,58 | 0,68 | 0,85 | 740 |
| 3 passages, sans reranking | 0,86 | 0,64 | 0,70 | 0,84 | 504 |
| **3 passages, avec reranking MiniLM** | **0,89** | 0,63 | **0,72** | **0,87** | 504 |

### Ce que montrent les résultats

1. **3 passages suffisent** : 5 passages n'améliorent pas les réponses et coûtent 47 % de tokens en plus. Ils peuvent même distraire le modèle : sur une question de réglage, il répond « Amniotes » avec 3 passages et « Turtles » avec 5.
2. **Le modèle n'est plus le facteur limitant, c'est la recherche** : le taux de réponses justes suit de près le taux de questions pour lesquelles le bon passage est fourni (86 % contre 84 %, puis 89 % contre 87 %). Quand il a le bon passage, le modèle répond presque toujours juste.
3. **Le reranking se retrouve dans les réponses** : 3 bons passages fournis en plus, 3 réponses justes en plus, pour le même nombre de tokens.
4. **Le modèle refuse de répondre quand il le doit**, mais pas toujours : sur une question de réglage où le bon passage manquait, il a inventé une réponse à partir de passages hors sujet au lieu de répondre « NOT FOUND ».

### Limites connues

- **La mesure peut se tromper sur les réponses très courtes.** Pour « How many municipalities are within Oberland? » (réponse : 6), le premier passage trouvé dit « 11 municipalities… five of them fall within Unterland, and the remainder within Oberland » : il contient la réponse, mais pas le chiffre 6.
- **Certaines références sont des phrases entières**, parfois mal alignées avec la question : le juge hésite sur ces cas.
- **Certains passages ont perdu leur sujet.** « The life expectancy is 82 years for women and 75 years for men » ne dit pas de quel pays il s'agit. Ajouter le titre de l'article devant chaque passage corrigerait ce défaut, mais le jeu de données ne fournit pas les titres.
- **Le juge est un LLM** : il peut se tromper, d'où la relecture à la main.

## Installation et reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # puis choisir la plateforme et mettre sa clé dans .env

pytest                             # 61 tests, sans réseau ni clé
python -m src.data                 # télécharge et sépare les données
python -m src.llm                  # vérifie la clé et le modèle (doit répondre « OK »)

# Étape 4 : recherche (aucun appel payant)
python -m src.eval_retrieval --questions reserve --rerank minilm bge

# Étape 5 : génération (appels au LLM)
python -m src.eval_generation                                              # 30 questions de réglage, 3 et 5 passages
python -m src.eval_generation --questions reserve --limit 100 --k 3 --rerank minilm
```

Le premier lancement de `eval_retrieval` calcule les vecteurs des passages, ce qui peut être long sur un ordinateur sans carte graphique. Les lancements suivants les relisent depuis `data/passage_vectors.npz`.

## Structure

```
src/
  config.py           paramètres (modèles, dossiers)
  data.py             téléchargement, nettoyage, séparation réglage / test / réserve
  embeddings.py       vecteurs avec BAAI/bge-base-en-v1.5
  retrieval.py        BM25, recherche dense, fusion RRF, reranking
  evaluation.py       Hit@k, MRR, réponse juste, F1 mots
  eval_retrieval.py   comparaison des méthodes de recherche
  llm.py              appel du LLM (toute plateforme compatible avec l'API d'OpenAI)
  generation.py       consignes, envoi des passages, lecture de la réponse et des sources
  judge.py            juge LLM
  eval_generation.py  évaluation des réponses
tests/                tests pytest (sans réseau)
notebooks/            exploration des données
results/              tableaux de résultats
data/                 données téléchargées (non versionnées, recréées par src.data)
```

## Outils

Python · rank_bm25 · snowballstemmer · sentence-transformers (embeddings et cross-encoder) · numpy · Hugging Face datasets · API OpenAI · pytest

## Références

- Yu et al. (2018), *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*.
- Cormack, Clarke et Büttcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*.
- Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering*.
- Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Comprehension of Text* (mesure F1 sur les mots).
- Nogueira et Cho (2019), *Passage Re-ranking with BERT* (principe du reranking par cross-encoder).
- Jeu de données [rag-datasets/rag-mini-wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) ; modèles [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) et [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
