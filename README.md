#### Set up project

**1. Install necessary libraries**

```bash
cd /path/to/the/project
python3 -m venv .venv
source .venv/bin/activate  # for MacOS
pip3 install -r requirements.txt
```

**2. Touch .env file**

```bash
touch .env
echo OPENAI_API_KEY=your_key > .env
```

#### Run project

**1. Set up Weaviate and Chunking**

```bash
docker compose up -d
python3 chunk.py
```

**2. Run the application**

```bash
python3 rag.py
```
