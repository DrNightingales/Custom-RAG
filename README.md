# continue-rag

A simple Retrieval-Augmented Generation (RAG) toolchain that:

1. **Indexes** your codebase (or any text files) into a local LanceDB vector database  
2. **Serves** a FastAPI endpoint to retrieve relevant code chunks via keyword-based full-text search

---

## Table of Contents

- [continue-rag](#continue-rag)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [A) Installing via pip (package mode)](#a-installing-via-pip-package-mode)
    - [B) Using a virtual environment (dev mode)](#b-using-a-virtual-environment-dev-mode)
  - [Configuration](#configuration)
    - [Environment variables (`.env`)](#environment-variables-env)
  - [Usage](#usage)
    - [1. Index your codebase](#1-index-your-codebase)
    - [2. Run the FastAPI server](#2-run-the-fastapi-server)
      - [a) Via console script](#a-via-console-script)
      - [b) Direct `uvicorn` invocation](#b-direct-uvicorn-invocation)
  - [Examples](#examples)

---

## Features

- **Chunk & embed** files in arbitrary directories (Python, C/C++, Java, web assets, text, logs, etc.)  
- Configurable **presets**, **include/exclude** filters & **hidden-file** support  
- Uses **OpenAI embeddings** + **LanceDB** for efficient vector storage & full-text indexing  
- Lightweight **FastAPI** server with a `/retrieve` endpoint  

---

## Prerequisites

- Python >= 3.12  
- An OpenAI API key  
- (Optional) HTTP/HTTPS proxy settings if you are behind a firewall  

---

## Installation

### A) Installing via pip (package mode)

1. Clone the repo and `cd` into it  
2. (Optional) Create & activate a venv:  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   ```  
3. Install your project:  
   ```bash
   pip install .
   ```  
4. Two console scripts will be available:
   - `continue-rag-index` → runs the indexer  
   - `continue-rag-server` → runs the FastAPI server  

### B) Using a virtual environment (dev mode)

1. Create & activate a venv:  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Invoke the Python scripts in place (no install step):

   - Indexer: `python index_code.py …`  
   - Server:   `uvicorn server:app --reload …`  

---

## Configuration

### Environment variables (`.env`)

Create a file named `.env` in the project root with at least:

```dotenv
# Required to talk to OpenAI
OPENAI_API_KEY=sk-…

# Optional HTTP/HTTPS proxy (if needed)
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080
```

The indexer and server will pick these up automatically via `python-dotenv`.

---

## Usage

### 1. Index your codebase

```bash
continue-rag-index \
  --src-dir /path/to/your/project \
  --preset python,web \
  --include-exts md,json \
  --exclude-dirs .venv,build \
  --embedding-model text-embedding-3-large \
  [--include-hidden] \
  [--proxy-http http://…] \
  [--proxy-https http://…]
```

Key flags:

- `--src-dir` / `-D`  Directory to scan (defaults to `.`)  
- `--preset` / `-P`  Comma-list of presets: `python`, `c_cpp`, `java`, `web`, `default`  
- `--include-exts` / `-I` Extra extensions (e.g. `md,json`)  
- `--exclude-dirs` / `-E` Dirs to skip (e.g. `venv,build`)  
- `--include-hidden` / `-H`  Include dot-files & dirs  
- `--embedding-model` / `-M`  OpenAI embedding model (default: `text-embedding-3-large`)  

Folders named `db/` will be created under `--src-dir` automatically.

### 2. Run the FastAPI server

#### a) Via console script

```bash
continue-rag-server \
  --db-path /path/to/your/project/db \
  --host 0.0.0.0 \
  --port 8000
```

#### b) Direct `uvicorn` invocation

```bash
uvicorn server:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

The server exposes one endpoint:

```
POST /retrieve
Content-Type: application/json

{
  "query": "... your search text ...",
  "fullInput": "... optional longer context ..."
}
```

Responds with up to 25 matching code chunks:
```json
[
  {
    "name": "src/foo.py",
    "description": "src/foo.py",
    "content": "def foo():\n    return 'bar'"
  },
  …
]
```

---

## Examples

1. **Index** a repo and then **retrieve** from Python:

   ```bash
   continue-rag-index -D . -P python,web
   continue-rag-server -D ./db -H
   ```

   ```python
   import httpx

   resp = httpx.post(
       "http://127.0.0.1:8000/retrieve",
       json={"query": "database connect", "fullInput": None}
   )
   print(resp.json())
   ```

2. **Use proxies**:

   ```bash
   continue-rag-index -D . -P python \
     --proxy-http http://127.0.0.1:8888
   ```
