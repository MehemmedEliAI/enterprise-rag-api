# ⚡ Enterprise RAG System

A production-grade, end-to-end Enterprise Retrieval-Augmented Generation (RAG) platform built with 100% open-source components, free-tier cloud APIs, and full Docker containerization.

---

## 🌟 Key Features

- **Vector Search:** High-performance dense retrieval using **Qdrant** and `BAAI/bge-small-en-v1.5` embeddings.
- **Neural Re-ranking:** Precision context filtering using **FlashRank** (`ms-marco-TinyBERT-L-2-v2`).
- **Low-Latency Caching:** Instant response retrieval for duplicate queries powered by **Redis**.
- **High-Speed Inference:** Ultra-fast async LLM generation using the **Groq API** (`openai/gpt-oss-20b`).
- **Async Microservice Architecture:** Powered by **FastAPI** with structured request validation.
- **Interactive UI:** Dashboard built with **Streamlit** featuring real-time health checks, latency monitoring, and document management.
- **One-Command Containerization:** Fully orchestrated multi-container deployment via **Docker Compose**.

---

## 🏗️ Architecture Flow

```text
[ Document / PDF ] ──► [ Ingestion Service ] ──► [ Chunk & Embed (bge-small) ] ──► [ Qdrant Vector DB ]
                                                                                         │
[ Streamlit UI ]   ──► [ FastAPI /query ]    ──► [ Redis Cache Check ] ──────────────────┤
                             │                          │ (Miss)                         ▼
                             │                          └──► [ Vector Search & FlashRank Re-rank ]
                             │                                           │
                             │                                           ▼
                             └───────────────────────────────► [ Groq LLM Inference ]
```

---

## 🛠️ Tech Stack

| Component        | Technology                          | Description                              |
| ---------------- | ----------------------------------- | ---------------------------------------- |
| API Framework    | FastAPI                             | High-performance async REST API          |
| User Interface   | Streamlit                           | Full-featured interactive dashboard      |
| Vector Database  | Qdrant                              | Vector store for document embeddings     |
| Cache Store      | Redis                               | In-memory key-value cache                |
| Embedding Model  | `BAAI/bge-small-en-v1.5`            | Dense text vectorization                 |
| Re-ranker        | `ms-marco-TinyBERT-L-2-v2`          | Lightweight cross-encoder re-ranking     |
| LLM Provider     | Groq API (`openai/gpt-oss-20b`)     | Ultra-fast inference engine              |
| Containerization | Docker & Docker Compose             | Multi-service stack management           |

---

## 📂 Project Structure

```text
enterprise-rag-api/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── core/
│   │   └── config.py            # Environment settings & model config
│   └── services/
│       ├── ingestion_service.py # Text extraction & vector indexing
│       ├── retrieval_service.py # Vector search & FlashRank re-ranking
│       ├── cache_service.py     # Redis caching logic
│       ├── llm_service.py       # Groq API integration
│       └── rag_orchestrator.py  # End-to-end RAG pipeline
├── streamlit_app.py             # Streamlit frontend UI
├── Dockerfile.api               # Container spec for FastAPI backend
├── Dockerfile.ui                # Container spec for Streamlit frontend
├── docker-compose.yml           # Multi-container orchestration stack
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables (Git-ignored)
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-api.git
cd enterprise-rag-api
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
APP_NAME="Enterprise Multimodal RAG API"
ENVIRONMENT="development"
DEBUG=True

# Groq Credentials
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL_NAME="openai/gpt-oss-20b"

# Vector Store
QDRANT_HOST="qdrant"
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME="enterprise_docs_en"

# Models
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME="ms-marco-TinyBERT-L-2-v2"

# Cache
REDIS_URL="redis://redis:6379/0"
```

### 3. Launch with Docker Compose

Run the entire stack with a single command:

```bash
docker-compose up --build -d
```

---

## 🖥️ Application Access Points

Once the containers are running:

| Service                        | URL                                   |
| ------------------------------ | ------------------------------------- |
| Streamlit Web Dashboard        | http://localhost:8501                 |
| FastAPI Interactive Docs       | http://localhost:8000/docs            |
| Qdrant Web Console             | http://localhost:6333/dashboard       |

---

## 📡 API Endpoints

### `POST /api/v1/ingest`

Uploads and indexes a document (`.pdf` or `.txt`) into Qdrant.

- **Request:** `multipart/form-data` with a `file` payload.

### `POST /api/v1/query`

Executes vector retrieval, re-ranking, cache checking, and LLM answer generation.

**Request example:**

```json
{
  "query": "What are the main findings in the document?",
  "top_k": 3
}
```

**Response example:**

```json
{
  "query": "What are the main findings in the document?",
  "answer": "The document highlights three key operational priorities...",
  "contexts": ["...extracted text chunk 1...", "...extracted text chunk 2..."],
  "cached": false
}
```

---

## 🧹 Housekeeping & Logs

Follow container logs:

```bash
docker-compose logs -f
```

Stop all services:

```bash
docker-compose down
```

Clear the Redis cache manually:

```bash
docker exec -it enterprise_rag_redis redis-cli FLUSHALL
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
