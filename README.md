# PhilAI - Oncul Analitik Felsefe RAG

CLI-first Turkish philosophy assistant built on:

- Async web scraping from `onculanalitikfelsefe.com`
- Local text corpus (`oncul_dump/`)
- Pure Dense Retrieval (FAISS vector index)
- Agentic interaction modes (Chat, Debate, Arena)

For full architecture and technical details, see `TECHNICAL.md`.

## Quick Start

This project uses `uv` for dependency management.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Verify installation**:
    ```bash
    uv run main.py doctor
    ```

3.  **Run the application**:
    ```bash
    uv run main.py chat
    ```

## Core Commands

All commands can be run via `uv run main.py <command>`.

### Data Pipeline
```bash
# Scrape new articles and update index
uv run main.py sync

# Or run steps manually:
# uv run main.py scrape
# uv run main.py index
```

### Modes
```bash
# Agentic Chat (Smart routing + Multi-query)
uv run main.py chat

# Agentic Debate (Argument analysis + Counter-RAG)
uv run main.py debate --category "Din_Felsefesi"

# Argument Mapper (Recursive Tree)
uv run main.py map "Kötülük Problemi" --depth 3 --branching 3

# AI Arena (Two AIs debating each other)
uv run main.py arena

# Single shot question
uv run main.py ask "Epistemoloji nedir?"
```

## Retrieval Flags

Supported flags for `chat`, `debate`, `ask`:

- `--from YYYY-MM-DD`: Filter by start date.
- `--to YYYY-MM-DD`: Filter by end date.
- `--category <name>`: Filter by specific category.
- `--kategori`: Interactive category picker.
- `--otokategori`: Enable zero-shot semantic category routing (default in some modes).

## Evaluation

To run benchmarks:

```bash
uv run main.py eval --sample 30 --k 5
```

Current Baseline (Feb 2026):
- `Hit@5`: ~0.93
- `MRR`: ~0.78

## Project Structure

- `main.py`: Entry point
- `scrape.py`: Scraper logic
- `rag/`: RAG pipeline components
  - `indexer.py`: Chunking & Indexing
  - `retriever.py`: Search logic (Dense + MMR)
  - `agents.py`: LLM Agent logic
  - `chat.py`: Interaction loops
  - `config.py`: Configuration


