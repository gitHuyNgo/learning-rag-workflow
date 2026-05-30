# Learning RAG Workflow

A hands-on Retrieval-Augmented Generation (RAG) pipeline that ingests PDF documents, stores vector embeddings in Weaviate, and answers questions using a multi-stage retrieval and reranking strategy.

**Stack:** Weaviate · OpenAI (embeddings + LLM) · Docling · LlamaIndex · BM25 · BGE Reranker

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (chunk.py)"]
        PDF["PDF files<br/><i>test_data/*.pdf</i>"]
        Docling["Docling<br/>PDF → Markdown"]
        Parser["LlamaIndex<br/>MarkdownNodeParser"]
        EmbedIn["OpenAI Embeddings<br/>text-embedding-ada-002"]
        Weaviate[("Weaviate<br/>Vector Store")]
    end

    subgraph Query["Query (rag.py)"]
        User(["User Question"])
        EmbedQ["OpenAI Embeddings<br/>Query Vector"]
        VectorSearch["Vector Search<br/>top_k = 10"]
        BM25["BM25 Rerank<br/>top_k = 5"]
        BGE["BGE Reranker<br/>top_k = 3"]
        LLM["OpenAI GPT<br/>Answer Generation"]
        Answer(["Answer"])
    end

    PDF --> Docling --> Parser --> EmbedIn --> Weaviate
    User --> EmbedQ --> VectorSearch
    Weaviate --> VectorSearch
    VectorSearch --> BM25 --> BGE --> LLM --> Answer
```

---

## Ingestion Pipeline (chunk.py)

Run once (or whenever documents change) to parse PDFs, chunk them, embed each chunk, and load vectors into Weaviate.

```mermaid
flowchart LR
    subgraph Input
        A["test_data/*.pdf"]
    end

    subgraph Processing
        B["DocumentConverter<br/>(Docling)"]
        C["export_to_markdown()"]
        D["MarkdownNodeParser<br/>(LlamaIndex)"]
        E["OpenAIEmbedding<br/>.get_text_embedding()"]
    end

    subgraph WeaviateSetup
        F{"Collection<br/>exists?"}
        G["Delete & recreate<br/>document_chunk_embedding"]
        H["Create collection<br/>properties: text, source<br/>vector: self_provided"]
    end

    subgraph Storage
        I[("Weaviate<br/>Insert chunk + vector")]
    end

    A --> B --> C --> D
    D -->|"For each node"| E --> I
    F -->|Yes| G --> H
    F -->|No| H
    H --> A
```

**What happens step by step:**

1. **Connect to Weaviate** — local instance at `localhost:8080`.
2. **Reset collection** — drops `document_chunk_embedding` if it already exists, then recreates it with `text`, `source`, and client-supplied vectors (`DEFAULT_VECTORIZER_MODULE: none` in Docker).
3. **For each PDF in `test_data/`:**
   - Docling converts the PDF to Markdown.
   - LlamaIndex splits Markdown into nodes (chunks).
   - OpenAI embeds each chunk.
   - Chunk text, source filename, and vector are inserted into Weaviate.

---

## Query Pipeline (rag.py)

Interactive CLI loop: embed the question, retrieve candidates, rerank in two stages, then generate an answer with context.

```mermaid
flowchart TB
    Q(["User types question"])

    subgraph Stage1["Stage 1 — Vector Retrieval"]
        E1["Embed query<br/>OpenAIEmbedding"]
        VS["near_vector search<br/>limit = 10"]
        D1["10 candidate chunks<br/>with distance metadata"]
    end

    subgraph Stage2["Stage 2 — BM25 Rerank"]
        T1["Tokenize corpus + query"]
        BM25["BM25Okapi scoring"]
        D2["Top 5 chunks<br/>by lexical relevance"]
    end

    subgraph Stage3["Stage 3 — BGE Rerank"]
        Pairs["Build (query, chunk) pairs"]
        BGE["FlagReranker<br/>BAAI/bge-reranker-large"]
        D3["Top 3 chunks<br/>by semantic relevance"]
    end

    subgraph Stage4["Stage 4 — Generation"]
        CTX["Format context<br/>[Chunk N] Source + Content"]
        SYS["System prompt<br/>context-grounded assistant"]
        LLM["OpenAIResponses<br/>gpt-5.1"]
        OUT(["Printed answer"])
    end

    Q --> E1 --> VS --> D1
    D1 --> T1 --> BM25 --> D2
    D2 --> Pairs --> BGE --> D3
    D3 --> CTX --> SYS --> LLM --> OUT
```

**Why three retrieval stages?**

| Stage | Method | Role |
|-------|--------|------|
| Vector search | Cosine similarity on embeddings | Broad semantic recall — finds conceptually related chunks |
| BM25 | Lexical keyword matching | Filters noise; boosts chunks that share terms with the query |
| BGE reranker | Cross-encoder scoring | Fine-grained (query, passage) relevance before sending context to the LLM |

Only the final **3 chunks** are passed to the model, keeping the prompt focused and within context limits.

---

## System Components

```mermaid
flowchart TB
    subgraph Local["Your machine"]
        Chunk["chunk.py<br/>Ingestion script"]
        RAG["rag.py<br/>Query CLI"]
        Venv["Python 3.10+<br/>+ requirements.txt"]
        Data["test_data/<br/>PDF documents"]
    end

    subgraph Docker["Docker Compose"]
        WV[("Weaviate 1.30<br/>:8080 HTTP<br/>:50051 gRPC")]
        Vol[("weaviate_data<br/>persistent volume")]
        WV --- Vol
    end

    subgraph External["External APIs"]
        OAI["OpenAI API<br/>Embeddings + Chat"]
    end

    subgraph Models["Downloaded at runtime"]
        BGE["BAAI/bge-reranker-large<br/>(FlagEmbedding)"]
    end

    Data --> Chunk
    Chunk --> WV
    Chunk --> OAI
    RAG --> WV
    RAG --> OAI
    RAG --> BGE
    Venv --> Chunk
    Venv --> RAG
```

---

## Prerequisites

- **Python 3.10+**
- **Docker** (for Weaviate)
- **OpenAI API key** (embeddings + chat completions)

Place PDF files in a `test_data/` directory at the project root before running ingestion.

---

## Setup & Usage

### 1. Create a virtual environment and install dependencies

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Start Weaviate and ingest documents

```bash
docker compose up -d
python3 chunk.py    # macOS / Linux
python chunk.py     # Windows
```

### 4. Run the RAG CLI

```bash
python3 rag.py      # macOS / Linux
python rag.py       # Windows
```

Type a question at the prompt. Enter `exit` to quit.

---

## Project Structure

```
learning-rag-workflow/
├── chunk.py              # PDF ingestion → Weaviate
├── rag.py                # Interactive Q&A with reranking
├── docker-compose.yml    # Weaviate service
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── test_data/            # Place PDF files here (create if missing)
```

---

## Configuration Reference

| Setting | Location | Default |
|---------|----------|---------|
| Weaviate collection | `chunk.py`, `rag.py` | `document_chunk_embedding` |
| Vector retrieval `top_k` | `rag.py` | 10 |
| BM25 rerank `top_k` | `rag.py` | 5 |
| BGE rerank `top_k` | `rag.py` | 3 |
| LLM model | `rag.py` | `gpt-5.1` |
| Reranker model | `rag.py` | `BAAI/bge-reranker-large` |
| Weaviate port | `docker-compose.yml` | 8080 |
