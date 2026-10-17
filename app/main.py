from fastapi import FastAPI

from app.routes_aggregation import router as aggregation_router
from app.routes_export import router as export_router
from app.routes_ingestion import router
from app.state import AppState

state = AppState()

app = FastAPI(title="RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking")
app.include_router(router)
app.include_router(aggregation_router)
app.include_router(export_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to " + "RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking"}


@app.on_event("shutdown")
def _shutdown() -> None:
    state.close()
