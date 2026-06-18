# MedicalDataRAG_NoSQL_project (Qdrant + MTSamples )

Recherche sémantique sur environ 5 000 comptes rendus cliniques anonymisés
à l'aide de **Qdrant** (base de données vectorielle) et de
**sentence-transformers** (`all-MiniLM-L6-v2`, plongements de 384 dimensions).

## Arborescence du projet

```
qdrant_project/
├── docker-compose.yml   # conteneur Qdrant (ports 6333 / 6334)
├── requirements.txt     # dépendances Python
├── ingest.py            # CSV → plongements → Qdrant
├── query.py             # 3 requêtes de démonstration (CLI)
├── app.py               # interface Streamlit avec filtres
└── mtsamples.csv        # jeu de données (à télécharger manuellement, voir plus bas)
```

## 1. Démarrer Qdrant

Depuis le dossier du projet :

```bash
docker compose up -d
```

Vérifier le tableau de bord : <http://localhost:6333/dashboard>

## 2. Télécharger le jeu de données

Le jeu de données est disponible sur Kaggle :
<https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions>
(licence CC0, environ 13 Mo).

Télécharger `mtsamples.csv` et le placer dans ce dossier.

## 3. Installer les dépendances Python

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux / Mac
pip install -r requirements.txt
```

Lors du premier lancement, le modèle d'embedding (environ 80 Mo) sera
téléchargé automatiquement dans le cache HuggingFace.

## 4. Ingérer les données

```bash
python ingest.py
```

Sortie attendue (extrait) :

```
Raw rows: 4999
After cleaning: 4966 rows, 40 medical specialties
Connecting to Qdrant...
Created collection 'mtsamples' (dim=384, cosine)
Loading embedding model 'all-MiniLM-L6-v2'...
Embedding + upserting: 100%|██████████| 78/78
Done. Points in collection: 4966
Open the dashboard: http://localhost:6333/dashboard
```

On peut alors explorer la collection depuis le dashboard
(*Collections → mtsamples → Visualize*).

## 5. Exécuter les requêtes de démonstration

```bash
python query.py
```

Trois requêtes sont exécutées :

| # | Requête | Filtre | Objectif |
|---|---------|--------|----------|
| 1 | « chest pain and shortness of breath when climbing stairs » | aucun | sémantique pure — doit ramener des comptes rendus de cardiologie alors que la requête utilise un vocabulaire profane (ni « angina », ni « dyspnea ») |
| 2 | identique à la Q1 | `medical_specialty == "Cardiovascular / Pulmonary"` | recherche hybride : similarité vectorielle + filtre structuré |
| 3 | « severe recurrent headaches with visual disturbances and nausea » | aucun | thème différent — doit ramener des comptes rendus de neurologie |

## 6. Lancer l'interface Streamlit (recommandée pour la démonstration)

```bash
streamlit run app.py
```

Un onglet de navigateur s'ouvre sur <http://localhost:8501> avec :

- un panneau latéral indiquant l'état de la connexion, le nombre de
  documents indexés, un **filtre multi-sélection par spécialité médicale**,
  un curseur top-K et un curseur de score minimal ;
- un panneau principal avec une zone de saisie de requête, un ensemble
  d'exemples pré-rédigés (un clic remplit la zone) et un bouton de
  recherche ;
- une carte par résultat affichant le score cosinus, la spécialité, le
  nom du document, la description et un extrait dépliable de la
  transcription.

L'interface utilise le même client Qdrant et le même modèle d'embedding que
`query.py` ; seule la présentation diffère.

## Étapes suivantes

- Ajouter une CLI interactive pour saisir les requêtes en continu
  (`argparse`)
- Tester un modèle de domaine médical
  (par ex. `pritamdeka/S-PubMedBert-MS-MARCO`)
  et comparer la qualité du rappel
- Mesurer le temps d'indexation et la latence des requêtes
  (variables pour le chapitre comparatif du rapport)
- Régler les paramètres HNSW (`m`, `ef_construct`) et observer
  le compromis rappel / vitesse