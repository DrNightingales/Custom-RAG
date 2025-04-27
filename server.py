# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import lancedb
from rake_nltk import Rake
app   = FastAPI()
db    = lancedb.connect("db")
table = db.open_table("code_chunks")
table.create_fts_index(["text", "filename"], replace=True)

class ContinueQuery(BaseModel):
    query: str
    fullInput: str | None = None   # Continue sends this field too

@app.post("/retrieve")
async def retrieve(q: ContinueQuery):
    r = Rake()
    r.extract_keywords_from_text(q.fullInput)
    keywors = r.get_ranked_phrases()
    print(keywors)
    hits = table.search(" ".join(keywors)).limit(25).to_list()
    items = [
        {
            "name": h["filename"],
            "description": h["filename"],
            "content": h["text"],
        } for h in hits
    ]
    print(items)
    return items
