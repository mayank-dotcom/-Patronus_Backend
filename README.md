<div align="center">

# ⬛ NEXUS — Agentic Knowledge Discovery System

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-v4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Django-5-092E20?style=for-the-badge&logo=django&logoColor=white" />
  <img src="https://img.shields.io/badge/Django_Ninja-REST-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-default-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-ReAct_Agent-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-Atlas_Vector_Search-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/pdfplumber-Table_Aware-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" />
</p>

> **Transform static PDFs into a queryable, reasoning-capable knowledge base.**  
> Upload a document → Ask anything → Get precise, cited, table-aware answers powered by a ReAct AI agent.

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Data Flow](#-data-flow)
- [Agent Reasoning Flow](#-agent-reasoning-flow)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)
- [Setup & Installation](#-setup--installation)
- [How It Works](#-how-it-works)

---

## 🧠 Overview

**Nexus** is a full-stack Agentic RAG (Retrieval-Augmented Generation) system. It goes beyond simple Q&A — the AI agent *reasons*, *plans*, and *uses tools* to answer complex questions about your documents.

| Capability | Description |
|---|---|
| 📄 **PDF Ingestion** | Table-aware extraction using `pdfplumber`, preserving structured data |
| 🔍 **Vector Search** | Semantic similarity search via MongoDB Atlas Vector Search |
| 🤖 **ReAct Agent** | Multi-step reasoning: retrieves context, performs calculations, synthesizes answers |
| 🧮 **Math Tools** | Dedicated `CAGRCalculator` for precise financial computations |
| 💬 **Chat Memory** | Persisted chat history (localStorage + SQLite) with context window |
| 📊 **Markdown Output** | AI responses rendered with tables, bold highlights, and page citations |

---

## 🏗 System Architecture

```mermaid
graph TB
    subgraph FE["🖥️ Frontend — Next.js 16 (Port 3000)"]
        UI["App UI\n(page.tsx)"]
        CHAT["ChatInterface.tsx"]
        UPLOAD["FileUploader.tsx"]
        PDFV["PdfViewer.tsx"]
        LS["localStorage\n(Chat History)"]
    end

    subgraph BE["⚙️ Backend — Django + Ninja (Port 8000)"]
        API["Django Ninja API\n(/api/*)"]
        ETL["ETL Pipeline\n(etl.py)"]
        AGENT["ReAct Agent\n(agent.py)"]
        DB["SQLite DB\n(PDF records + Chat logs)"]
        MEDIA["Media Storage\n(/media/*.pdf)"]
    end

    subgraph AI["🤖 AI / External Services"]
        OAI_EMB["OpenAI\ntext-embedding-3-small"]
        OAI_LLM["OpenAI\nGPT-4o"]
        MONGO["MongoDB Atlas\nVector Search"]
    end

    UI --> CHAT
    UI --> UPLOAD
    UI --> PDFV
    CHAT <-->|"POST /api/chat\nDELETE /api/chat/clear"| API
    UPLOAD -->|"POST /api/upload"| API
    UI <-->|"GET /api/pdfs"| API

    API --> ETL
    API --> AGENT
    API --> DB
    ETL --> MEDIA
    ETL --> OAI_EMB
    OAI_EMB --> MONGO
    AGENT --> OAI_LLM
    AGENT <-->|"Similarity Search"| MONGO
    CHAT <--> LS
```

---

## 🔄 Data Flow

### PDF Upload & Indexing

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant API as Django API
    participant ETL as ETL Pipeline
    participant OAI as OpenAI Embeddings
    participant MDB as MongoDB Atlas

    User->>FE: Drag & drop PDF
    FE->>API: POST /api/upload (multipart)
    API->>ETL: process_pdf(path, config)
    ETL->>ETL: pdfplumber extracts text + tables
    ETL->>ETL: Convert tables → Markdown
    ETL->>ETL: Build Document chunks (1 per page)
    ETL->>OAI: text-embedding-3-small (1536-dim)
    OAI-->>ETL: Embedding vectors
    ETL->>MDB: Store docs + vectors
    MDB-->>ETL: Confirmed
    ETL-->>API: "Processed N pages"
    API-->>FE: { message: "Success" }
    FE-->>User: ✅ Upload confirmed, PDF available in Archives
```

### Chat Query & Agent Response

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant API as Django API
    participant AGENT as ReAct Agent
    participant KB as KnowledgeBase Tool
    participant CAGR as CAGRCalculator Tool
    participant MDB as MongoDB Atlas
    participant LLM as GPT-4o

    User->>FE: Types a question
    FE->>API: POST /api/chat { query, chat_history }
    API->>AGENT: invoke({ input, chat_history })

    loop ReAct Loop (up to 20 iterations)
        AGENT->>LLM: Thought → Action decision
        alt Needs document context
            AGENT->>KB: retrieve_info(query)
            KB->>MDB: similarity_search(k=6)
            MDB-->>KB: Top-6 chunks with page numbers
            KB-->>AGENT: "[Page X]: ..."
        else Needs calculation
            AGENT->>CAGR: calculate_cagr("baseline, target, years")
            CAGR-->>AGENT: "CAGR is X.XX%"
        end
        AGENT->>LLM: Observation → Next Thought
    end

    AGENT-->>API: Final Answer + intermediate_steps
    API->>API: Store Q+A in SQLite
    API-->>FE: { answer, thought_process }
    FE-->>User: Rendered Markdown response + [TRACE LOG]
```

---

## 🤖 Agent Reasoning Flow

```mermaid
flowchart TD
    Q(["❓ User Question"]) --> THINK["🧠 LLM: Think\nWhat do I need?"]
    THINK --> DECIDE{Action?}

    DECIDE -->|"Need document data"| KB["🔍 KnowledgeBase Tool\nMongoDB Vector Search k=6"]
    DECIDE -->|"Need calculation"| MATH["🧮 CAGRCalculator Tool\nbaseline, target, years → CAGR%"]
    DECIDE -->|"Have enough context"| FINAL

    KB --> OBS1["📋 Observation\nPage-cited chunks"]
    MATH --> OBS2["📊 Observation\nCGAR = X.XX%"]

    OBS1 --> LOOP{More\nsteps needed?}
    OBS2 --> LOOP

    LOOP -->|Yes| THINK
    LOOP -->|No| FINAL["✅ Final Answer\nMarkdown + Page Citations"]
    FINAL --> OUT(["💬 Rendered Response"])
```

---

## 📁 Project Structure

```
nexus/
├── 📄 README.md
│
├── backend/                        # Django backend
│   ├── core/
│   │   ├── settings.py             # Django settings, DB config, CORS
│   │   ├── urls.py                 # Root URL router
│   │   ├── wsgi.py / asgi.py
│   │   └── __init__.py
│   │
│   ├── api/
│   │   ├── api.py                  # ⭐ All API endpoints (upload, chat, pdfs)
│   │   ├── models.py               # UploadedPDF, ChatMessageStored models
│   │   ├── admin.py
│   │   ├── migrations/             # DB migrations
│   │   └── utils/
│   │       ├── etl.py              # ⭐ PDF extraction + vector ingestion pipeline
│   │       └── agent.py            # ⭐ ReAct Agent (GPT-4o + tools)
│   │
│   ├── media/                      # Uploaded PDFs (served at /media/)
│   ├── db.sqlite3                  # SQLite DB (PDF records + chat logs)
│   ├── manage.py
│   ├── requirements.txt
│   ├── .env                        # 🔒 Secret keys (not in git)
│   └── .env.example                # ✅ Template for env setup
│
└── frontend/                       # Next.js 16 frontend
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx            # ⭐ Main layout (sidebar + chat + pdf viewer)
    │   │   ├── layout.tsx          # Root layout
    │   │   └── globals.css         # Global styles + Tailwind v4
    │   │
    │   └── components/
    │       ├── ChatInterface.tsx   # ⭐ Chat UI with ReAct trace log
    │       ├── FileUploader.tsx    # Drag-and-drop PDF uploader
    │       └── PdfViewer.tsx       # In-browser PDF preview panel
    │
    ├── public/                     # Static assets
    ├── .env.local                  # 🔒 Frontend env (API URL)
    ├── .env.example                # ✅ Template for frontend env
    ├── next.config.ts
    ├── package.json
    ├── postcss.config.mjs
    └── tsconfig.json
```

---

## 🔌 API Reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description | Body / Params |
|--------|----------|-------------|---------------|
| `POST` | `/api/upload` | Upload a PDF for processing & indexing | `multipart/form-data` — `file` |
| `GET` | `/api/pdfs` | List all uploaded PDFs | — |
| `POST` | `/api/chat` | Send a query to the ReAct agent | `{ query: string, chat_history: [] }` |
| `DELETE` | `/api/chat/clear` | Clear all stored chat messages | — |

<details>
<summary><strong>POST /api/chat — Request & Response Example</strong></summary>

**Request**
```json
{
  "query": "What is the total number of jobs reported and where is it stated?",
  "chat_history": [
    { "role": "human", "content": "Hello" },
    { "role": "ai", "content": "Hi! Upload a PDF to get started." }
  ]
}
```

**Response**
```json
{
  "answer": "The report states **12,500 jobs** were created *(Page 4)*.",
  "thought_process": "Thought: I need to search for jobs data.\nAction: KnowledgeBase\nObservation: [Page 4]: Total jobs reported: 12,500..."
}
```
</details>

---

## 🔐 Environment Variables

### Backend — `backend/.env`

Copy `backend/.env.example` and fill in your values:

```bash
cp backend/.env.example backend/.env
```

| Variable | Description | Example |
|---|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string | `mongodb+srv://user:pass@cluster.mongodb.net/` |
| `DB_NAME` | MongoDB database name | `ReaderDb` |
| `COLLECTION_NAME` | MongoDB collection for vectors | `reader_collection` |
| `VECTOR_INDEX_NAME` | Atlas Vector Search index name | `vector_index` |
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o + embeddings) | `sk-...` |
| `SECRET_KEY` | Django secret key | `django-insecure-...` |
| `DEBUG` | Django debug mode | `True` / `False` |

### Frontend — `frontend/.env.local`

Copy `frontend/.env.example` and fill in:

```bash
cp frontend/.env.example frontend/.env.local
```

| Variable | Description | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend base URL (no trailing slash) | `http://127.0.0.1:8000` |

---

## 🚀 Setup & Installation

### Prerequisites

| Tool | Minimum Version |
|---|---|
| Node.js | v20+ |
| Python | v3.11+ |
| pip | Latest |
| MongoDB Atlas | Free tier cluster with Vector Search index |
| OpenAI API Key | With access to `gpt-4o` and `text-embedding-3-small` |

---

### 1️⃣ Clone the Repository

```bash
git clone <repo-url>
cd nexus
```

---

### 2️⃣ Backend Setup

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# ✏️ Edit .env with your MONGODB_URI, OPENAI_API_KEY, etc.

# Run database migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

Backend will be live at → **http://127.0.0.1:8000**

---

### 3️⃣ Frontend Setup

```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# ✏️ Edit .env.local — set NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

# Start the development server
npm run dev
```

Frontend will be live at → **http://localhost:3000**

---

### 4️⃣ MongoDB Atlas — Vector Search Index Setup

In your MongoDB Atlas dashboard, create a Vector Search index on your collection with the following definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

> Index name must match `VECTOR_INDEX_NAME` in your `.env`.

---

## ⚙️ How It Works

### 1. PDF Ingestion (ETL)

- `pdfplumber` opens the PDF and processes **one page at a time**
- Plain text is extracted via `extract_text()`
- Tables are extracted via `extract_tables()` and converted to **Markdown** format — this preserves the header→value relationship critical for tabular Q&A
- Each page becomes a LangChain `Document` with `page_number` metadata
- Documents are embedded using **OpenAI `text-embedding-3-small`** (1536 dimensions)
- Vectors are stored in **MongoDB Atlas Vector Search**

### 2. ReAct Agent

The agent uses the **ReAct (Reasoning + Acting)** pattern via LangChain:

- **KnowledgeBase Tool** — semantic similarity search (`k=6`) in MongoDB Atlas, returns chunks with page citations
- **CAGRCalculator Tool** — pure Python math tool, takes `baseline, target, years` → returns precise `CAGR%`
- Agent iterates up to **20 steps**, with a **5-minute timeout**
- All intermediate reasoning steps (Thought → Action → Observation) are returned to the UI as a collapsible **TRACE LOG**

### 3. Chat Memory

- Frontend stores messages in **localStorage** (survives page refresh)
- Backend stores all messages in **SQLite** (`ChatMessageStored` model)
- Last **5 message exchanges** are passed as formatted `chat_history` to the agent on every request

---

<div align="center">
  <sub>Built with ❤️ · Next.js · Django · LangChain · MongoDB Atlas · OpenAI</sub>
</div>
