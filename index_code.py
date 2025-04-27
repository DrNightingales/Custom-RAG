# index.py
import os, pathlib, tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import openai
import lancedb, numpy as np
from tqdm import tqdm

load_dotenv()

# Configure your proxy
PROXIES = {
    'http': os.getenv('HTTP_PROXY'), 
    'https': os.getenv('HTTPS_PROXY')
}

MODEL = "text-embedding-3-large"                 # 3072-D by default
SRC_DIR = pathlib.Path("/home/drn/Dev/syllabus_server")# change to your codebase
EXCLUDED_DIRS={'__pycache__', 'venv'}
DB_DIR  = "db"
TABLE   = "code_chunks"
openai.proxy = PROXIES
client  = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    
)

tokenizer = tiktoken.encoding_for_model(MODEL)

def chunk(path: pathlib.Path, max_tokens=4096):
    """Very naive line-based chunker; good enough to start."""
    buf, count = [], 0
    for line in path.read_text(errors="ignore").splitlines():
        t = len(tokenizer.encode(line))
        if count + t > max_tokens:
            yield "\n".join(buf)
            buf, count = [], 0
        buf.append(line); count += t
    if buf: yield "\n".join(buf)

def index_codebase(table):
    """Index the codebase and store embeddings in LanceDB."""
    # Get all Python files in the source directory, excluding hidden files and directories
    extensions = ['py','js','html','css','xml']
    files = []
    for ext in extensions:
        glob = SRC_DIR.rglob(f'*.{ext}')
        ext_files = [
            f for f in glob
            if not any(part.startswith('.') for part in f.parts) 
            and not any(excluded in f.parts for excluded in EXCLUDED_DIRS)
        ]
        files.extend(ext_files)
        total_files = len(files)
    # Initialize tqdm progress bar
    with tqdm(total=total_files, desc="Indexing codebase", unit="file") as pbar:
        for f in files:
            for text in chunk(f):
                emb = client.embeddings.create(model="text-embedding-3-large", input=text).data[0].embedding
                table.add([{"filename": str(f), "text": text, "vector": np.array(emb)}])
            pbar.update(1)

if __name__ == "__main__":
    db  = lancedb.connect(DB_DIR)
    if TABLE in db.table_names():
        table = db.open_table(TABLE)
    else:
        from lancedb.pydantic import LanceModel, Vector
        class CodeChunk(LanceModel):
            filename: str
            text: str
            vector: Vector(3072)
        table = db.create_table(TABLE, schema=CodeChunk, mode="overwrite")
    index_codebase(table)