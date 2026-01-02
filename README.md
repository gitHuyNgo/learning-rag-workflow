# RAG Project Setup

This project implements a basic RAG pipeline using Weaviate to store vectors, OpenAI for embeddings, docling and llama-index for ingestion.

---

## Prerequisites

Make sure you have installed:

- **Python 3.10+**
- **Docker**
- **OpenAI API Key**

---

## Usage

### 1. Create a virtual environment & Install dependencies

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


### 2. Create ```.env``` file
Create a file named ```.env``` in the project root and add:
```bash
OPENAI_API_KEY=your_api_key
```


### 3. Start Weaviate and run chunking
```bash
docker compose up -d
python3 chunk.py    # For Mac
python chunk.py     # For Windows
```


### 4. Run the application
```bash
python3 rag.py      # For Mac
python rag.py       # For Windows
```