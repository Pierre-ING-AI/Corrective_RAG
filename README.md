# 🛡️ Corrective RAG (CRAG) avec LangGraph

Ce projet implémente un pipeline **RAG (Retrieval-Augmented Generation) correctif** basé sur **LangGraph**, combinant plusieurs techniques avancées de récupération d'information : **Multi-Query**, **Routing** (évaluation de pertinence des documents) et **Indexation classique** (chunk-based).

<div align="center">
  <img src="images/Capture d’écran 2026-08-07 171929.png" width="60%">
</div>

---

## 📖 Sommaire

- [🛡️ Corrective RAG (CRAG) avec LangGraph](#️-corrective-rag-crag-avec-langgraph)
  - [📖 Sommaire](#-sommaire)
  - [🎯 Contexte](#-contexte)
  - [🧩 Techniques implémentées](#-techniques-implémentées)
  - [🏗️ Architecture du graphe](#️-architecture-du-graphe)
  - [⚙️ Prérequis](#️-prérequis)
  - [📦 Installation](#-installation)
  - [🔑 Configuration](#-configuration)
  - [🚀 Utilisation](#-utilisation)
  - [📁 Structure du projet](#-structure-du-projet)
  - [🔍 Fonctionnement détaillé](#-fonctionnement-détaillé)
    - [1. Indexation](#1-indexation)
    - [2. Reformulation (Multi-Query)](#2-reformulation-multi-query)
    - [3. Récupération](#3-récupération)
    - [4. Jugement de pertinence (Routing)](#4-jugement-de-pertinence-routing)
    - [5. Décision conditionnelle](#5-décision-conditionnelle)
  - [📌 Notes](#-notes)

---

## 🎯 Contexte

Ce dépôt documente et implémente différentes stratégies avancées de RAG :

- **Query Translation** : Multi-Query, RAG Fusion, Decomposition, Step-Back Prompting, HyDE
- **Routing** : acheminement logique ou sémantique des requêtes
- **Query Construction** : conversion de langage naturel vers requêtes structurées (SQL, filtres de métadonnées, etc.)
- **Indexing** : indexation classique, résumés (Parent-Document), RAPTOR, ColBERT
- **Active CRAG** et **Adaptive RAG** via machines à états (`StateGraph`)

Le projet applicatif (`agent.py` / notebook) met en œuvre une version simplifiée du **Corrective RAG**, combinant :
1. **Multi-query** (reformulation de la question)
2. **Routing** (jugement de pertinence des documents récupérés — oui/non)
3. **Indexation classique** (chunking + recherche vectorielle avec ChromaDB)

---

## 🧩 Techniques implémentées

| Composant | Rôle |
| :--- | :--- |
| **Reformulateur (Multi-Query)** | Génère 5 reformulations de la question utilisateur via un LLM |
| **BDD vectorielle (retriever)** | Recherche les documents pertinents pour chaque reformulation dans ChromaDB, puis sélectionne les meilleurs par fréquence d'occurrence |
| **Juge (Grader)** | Évalue la pertinence (`yes`/`no`) de chaque document récupéré par rapport à la question, via une sortie structurée (Pydantic) |
| **Routage conditionnel** | Si aucun document n'est jugé pertinent → message d'erreur ; sinon → génération de la réponse |
| **LLM (génération)** | Génère la réponse finale en s'appuyant uniquement sur les documents jugés pertinents |
| **Message d'erreur** | Réponse de repli lorsque aucun document pertinent n'a été trouvé (garde-fou hors-sujet) |

---

## 🏗️ Architecture du graphe

```
START
  │
  ▼
Reformulateur (Multi-Query)
  │
  ▼
bdd_vectorielle (retriever)
  │
  ▼
Juge (grade_documents)
  │
  ├── documents pertinents ──► llm ──► END
  │
  └── aucun document pertinent ──► messageErreur ──► END
```

Le graphe est construit avec `langgraph.graph.StateGraph` et compilé en `agent`. L'état partagé (`AgentState`) circule entre les nœuds :

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    table_questions: list[str]
    documents: list[str]
    juge: str
```

---

## ⚙️ Prérequis

- Python 3.10+
- Une clé API OpenAI valide
- Les dépendances suivantes :

```
langchain
langchain-openai
langchain-core
langchain-chroma
langgraph
pydantic
python-dotenv
```

---

## 🔑 Configuration

Créez un fichier `.env` à la racine du projet avec votre clé API OpenAI :

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

Le fichier `documentOriginal.txt` doit être présent à la racine du projet : il constitue le corpus source qui sera découpé en paragraphes puis indexé dans ChromaDB (dossier `BDD/`, collection `histoire`).

---

## 🚀 Utilisation

Lancez le notebook ou le script principal :

```bash
python main.py
```

ou, dans un notebook Jupyter, exécutez les cellules dans l'ordre. Exemple d'appel :

```python
response = agent.invoke({"messages": [HumanMessage(content="Qui a envahi la Gaule ?")]})
print(response["messages"][-1].content)
```

- Si la question est **liée au corpus** (ex. histoire de la Gaule), l'agent retrouve les documents pertinents et génère une réponse sourcée.
- Si la question est **hors sujet** (ex. "Que font 5 × 5 ?"), aucun document pertinent n'est trouvé et l'agent renvoie un message de repli :
  > *"Je suis un assistant historien et je ne peux pas répondre à votre question."*

---

## 📁 Structure du projet

```
.
├── documentOriginal.txt       # Corpus source brut
├── document_modifie.txt       # Corpus découpé en paragraphes (généré)
├── BDD/                       # Base vectorielle ChromaDB persistée
├── images/
│   ├── Overview.png
│   ├── adaptiveRAG.png
│   └── Capture d'écran ....png
├── main.py / notebook.ipynb   # Script principal (construction du graphe)
├── .env                       # Variables d'environnement (clé API)
└── README.md
```

---

## 🔍 Fonctionnement détaillé

### 1. Indexation
Le corpus (`documentOriginal.txt`) est découpé en paragraphes (marqueur `Article détaillé`), puis chaque paragraphe est vectorisé avec `text-embedding-3-large` et stocké dans **ChromaDB** (persisté dans `BDD/`).

### 2. Reformulation (Multi-Query)
Le LLM (`gpt-4.1-mini`) génère **5 reformulations** de la question initiale afin de couvrir différentes formulations lexicales possibles.

### 3. Récupération
Chaque reformulation est envoyée séparément au retriever (`k=4`). Les documents obtenus sont agrégés, puis les **4 documents les plus fréquemment retrouvés** (via `Counter`) sont retenus.

### 4. Jugement de pertinence (Routing)
Chaque document sélectionné est évalué par le LLM avec une sortie structurée (`GradeDocuments`, score binaire `yes`/`no`) afin de déterminer s'il répond réellement à la question.

### 5. Décision conditionnelle
- Si **aucun document pertinent** n'est trouvé → le graphe route vers le nœud `messageErreur`.
- Sinon → le graphe route vers le nœud `llm`, qui génère la réponse finale à partir des documents filtrés.

---

## 📌 Notes

- Le prompt système contraint volontairement l'assistant à un rôle d'**historien**, ce qui explique le message de repli pour les questions hors-sujet.
- La sélection des "meilleurs documents" repose sur leur **fréquence d'apparition** parmi les résultats des différentes reformulations, une approche proche du principe de vote majoritaire (à distinguer du re-ranking RRF utilisé dans RAG Fusion).
- Ce projet peut être étendu avec les autres techniques documentées (RAG Fusion, HyDE, Step-Back, RAPTOR, ColBERT, Adaptive RAG) pour améliorer la robustesse du pipeline.
