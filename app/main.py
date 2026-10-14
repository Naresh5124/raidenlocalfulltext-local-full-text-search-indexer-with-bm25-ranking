from fastapi import FastAPI

app = FastAPI(title="RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking")

@app.get("/")
def read_root():
    return {"message": "Welcome to " + "RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking"}
