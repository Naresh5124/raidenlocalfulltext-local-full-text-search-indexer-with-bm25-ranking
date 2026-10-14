# RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking

RaidenLocalFullText: Local Full-Text Search Indexer with BM25 Ranking is a GitForge-generated project with a local-first development setup.

## Tech stack

- Python
- FastAPI
- SQLite

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload
```

## Verify

```bash
pytest
```

See `docs/` for architecture notes, decisions, and generated module documentation.
