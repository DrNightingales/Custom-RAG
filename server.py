import argparse
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
import lancedb  # type: ignore
from rake_nltk import Rake  # type: ignore

app = FastAPI()
_rake = Rake()

# Default, overridden by CLI arg when run as a script
DB_PATH: str = "db"


def get_code_chunks_table() -> lancedb.table:
    """Connect to LanceDB using the configured path and return the 'code_chunks' table."""
    db = lancedb.connect(DB_PATH)
    table = db.open_table("code_chunks")
    table.create_fts_index(["text", "filename"], replace=True)
    return table


class ContinueQuery(BaseModel):
    query: str
    fullInput: str | None = None


@app.post("/retrieve", response_model=list[dict])
async def retrieve(
    payload: ContinueQuery,
    table: lancedb.table = Depends(get_code_chunks_table),
) -> list[dict]:
    """
    Extract keywords from `fullInput` (or fall back to `query`),
    perform a full-text search, and return up to 25 matching chunks.
    """
    text_to_analyze = payload.fullInput or payload.query
    if not text_to_analyze.strip():
        raise HTTPException(status_code=400, detail="No input text provided")

    _rake.extract_keywords_from_text(text_to_analyze)
    keywords = _rake.get_ranked_phrases()
    search_terms = " ".join(keywords) or payload.query

    hits = table.search(search_terms).limit(25).to_list()
    return [
        {
            "name": hit["filename"],
            "description": hit["filename"],
            "content": hit["text"],
        }
        for hit in hits
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastAPI server with specified LanceDB path"
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help="Path to the LanceDB database directory",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server",
    )
    args = parser.parse_args()

    # Override global DB_PATH for dependency resolution
    DB_PATH = args.db_path  # type: ignore

    uvicorn.run(app, host=args.host, port=args.port)
