# Technical Documentation - PhilAI RAG System

## 1. System Overview

PhilAI is a local RAG pipeline for Turkish philosophy content.

Main flow:

1. Scrape articles from source website into text files.
2. Parse metadata and chunk text.
3. Build FAISS index for dense retrieval.
4. Run vector retrieval with MMR diversification and optional Reranker.
5. Feed context into LLM modes:
   - Chat: Smart routing + multi-query expansion + history.
   - Debate: Claim extraction + counter-search + contradiction analysis.
   - Arena: Dynamic RAG-based debate between two AI personas.

Entry point is `main.py`.

## 2. Repository Components

- `main.py`: CLI command router.
- `scrape.py`: async crawler and per-category file writer.
- `rag/config.py`: embedding/chunk/retrieval config.
- `rag/indexer.py`: document loading, chunking, embedding, index build/update.
- `rag/retriever.py`: retrieval engine (vector search, MMR, date/category filters, URL-unique centroids).
- `rag/agents.py`: lightweight LLM planners (routing, query expansion, claim extraction, contradiction analysis).
- `rag/chat.py`: chat/debate/arena orchestration and formatting.
- `rag/doctor.py`: health diagnostics.
- `rag/eval.py`: benchmark/evaluation harness.

## 3. Data Model ("Database")

There are two storage layers:

1. Raw corpus layer (`oncul_dump/`)
2. Retrieval index layer (`faiss_index/<provider_model>/`)

### 3.1 Raw Corpus

Each article is saved as `.txt` with header metadata (TITLE, URL, DATE, AUTHOR, CATEGORIES).

### 3.2 Retrieval Index

Stored artifacts: `index.faiss`, `chunks.pkl`, `metadatas.pkl`, `config.json`.

## 4. DB Health Snapshot (Last Update: 2026-02-14)

Measured via `python main.py doctor`:

### 8.1 Chat Mode

1. Decide CHAT vs RAG route (`should_use_rag`).
2. Expand query into multi-query set.
3. Run retrieval (`multi_search`) with current mode/filters.
4. Build context and call LLM.
5. Append evidence snippets mapped to cited `[Kaynak n]` markers.

### 8.2 Debate Mode

1. Extract claims from user argument.
2. Generate counter-search queries.
3. Retrieve counter-evidence.
4. Run contradiction analysis against retrieved context.
5. Inject structured debate notes into system context.

### 8.3 Arena Mode

- Two LLM personas debate opposing positions.
- Uses same retrieval primitives for topic grounding.

## 9. Benchmarking / Evaluation

Implemented in `rag/eval.py`.

### 9.1 Dataset

- Auto-built JSONL from index unique URLs/titles.
- Each sample contains:
  - `query`
  - `expected_urls`
  - optional `category`, `date_from`, `date_to`

### 9.2 Metrics

- `Hit@K`: expected URL appears in top K
- `MRR`: reciprocal rank quality
- `Category Top1 Acc`: semantic category suggestion top-1 accuracy

### 9.3 Current Baseline (2026-02-14)

From `python main.py eval --sample 30 --k 5 --mode hybrid`:

- Samples: `30`
- Hit@5: `0.800`
- MRR: `0.614`
- Category Top1 Acc: `0.433`

Notes:

- This is a small sample baseline, not a full offline benchmark.
- Increase sample size and add manually curated hard questions for stronger confidence.

## 10. Operational Commands

Data and index:

```bash
./.venv/bin/python main.py scrape
./.venv/bin/python main.py index
./.venv/bin/python main.py sync
```

Retrieval quality ops:

```bash
./.venv/bin/python main.py doctor
./.venv/bin/python main.py eval --sample 120 --k 5 --mode hybrid
./.venv/bin/python main.py categories "epistemoloji ve bilgi"
```

Interaction with advanced filters:

```bash
./.venv/bin/python main.py chat --mode hybrid --otokategori
./.venv/bin/python main.py ask "siyaset felsefesi 2010 sonrasi" --from 2010-01-01 --to 2020-12-31
```

Using `uv` (modern Python package manager):
```bash
uv run main.py sync        # Sync content + index
uv run main.py doctor      # Health check
uv run main.py eval        # Run benchmarks
uv run main.py chat        # Chat mode
uv run main.py debate      # Debate mode
uv run main.py arena       # Arena mode
```

Or activate virtualenv manually: `source .venv/bin/activate`

## 11. Development Setup

This project uses `uv` for dependency management.

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone repo & init: `uv sync`
3. Run: `uv run main.py ...`

Dependencies are tracked in `pyproject.toml` and locked in `uv.lock`.

## 12. Current Risks and Recommended Next Steps

1. Index rebuild (`uv run main.py index --full`).
2. Cap chunks per URL during indexing.
3. Hand-labeled eval dataset.
4. Better Turkish Reranker.
5. Add automated CI check to run:
   - `main.py doctor`
   - a fixed eval slice
   - regression thresholds
