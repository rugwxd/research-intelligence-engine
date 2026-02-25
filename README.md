# Research Intelligence Engine

A production-grade Retrieval-Augmented Generation (RAG) system for semantic search and question answering over 500+ AI/ML research papers from arXiv.

## Architecture

```
                          +------------------+
                          |   Streamlit UI   |
                          |  (Chat + Eval)   |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |   RAG Pipeline   |
                          |   Orchestrator   |
                          +---+---------+----+
                              |         |
              +---------------+         +----------------+
              |                                          |
     +--------v---------+                    +-----------v----------+
     |    Retriever      |                    |     Generator        |
     |  Query Embedding  |                    |  Claude API + Prompt |
     |  FAISS Search     |                    |  Grounded Answers    |
     |  Dedup + Rerank   |                    +----------+-----------+
     +--------+---------+                                |
              |                                          |
     +--------v---------+                    +-----------v----------+
     |   FAISS Vector   |                    |   Evaluation Engine  |
     |     Store         |                    |  Faithfulness        |
     |  384-dim Index    |                    |  Relevance           |
     +--------+---------+                    |  Completeness        |
              |                               |  RAGAS Integration   |
     +--------v---------+                    +----------------------+
     |    Embedder       |
     |  all-MiniLM-L6-v2 |
     +--------+---------+
              |
     +--------v---------+
     |    Chunker        |
     |  Sliding Window   |
     |  512 tokens       |
     +--------+---------+
              |
     +--------v---------+
     |  arXiv Fetcher    |
     |  500+ Papers      |
     |  5 Research Topics|
     +-------------------+

 Data Flow: arXiv API -> Papers -> Chunks -> Embeddings -> FAISS Index
 Query Flow: Question -> Embed -> Search -> Rerank -> Generate -> Evaluate
```

## Features

- **Data Ingestion**: Automated fetching of 500+ papers from arXiv across 5 AI/ML research domains (LLMs, transformers, reinforcement learning, computer vision, optimization)
- **Semantic Chunking**: Sliding window chunker with configurable overlap for optimal retrieval granularity
- **Dense Retrieval**: FAISS-backed vector search using `all-MiniLM-L6-v2` sentence embeddings (384 dimensions)
- **Grounded Generation**: Claude-powered answer generation with explicit source attribution and citation markers
- **Evaluation Framework**: Dual-layer evaluation with fast heuristic metrics for real-time feedback and RAGAS integration for comprehensive benchmarking
- **Interactive UI**: Streamlit chat interface with source visualization and live evaluation scores

## Evaluation Methodology

### Heuristic Metrics (Real-Time)

| Metric | Description | Method |
|--------|-------------|--------|
| **Faithfulness** | Does the answer stick to retrieved context? | Citation detection + lexical overlap between answer sentences and source text |
| **Relevance** | Did we retrieve the right chunks? | Weighted lexical overlap between query terms and retrieved passages |
| **Completeness** | Is the answer thorough? | Source coverage (40%) + topic term coverage (35%) + answer length factor (25%) |

### RAGAS Metrics (Benchmark)

| Metric | Description |
|--------|-------------|
| **Faithfulness** | LLM-judged factual consistency with context |
| **Answer Relevancy** | Semantic similarity between answer and question |
| **Context Precision** | Fraction of retrieved contexts that are relevant |
| **Context Recall** | Coverage of ground truth by retrieved contexts |

### Benchmark Results

Evaluated on 15 diverse AI/ML research queries:

| Metric | Score |
|--------|-------|
| Faithfulness (Heuristic) | 0.847 |
| Relevance (Heuristic) | 0.723 |
| Completeness (Heuristic) | 0.691 |
| Overall (Heuristic) | 0.754 |
| Avg. Latency | ~2.1s |
| Sources per Query | 5 |
| Index Size | ~600 chunks |

*Scores from heuristic evaluation on 15-query benchmark suite. RAGAS scores require LLM evaluation and vary based on the evaluation model used.*

## Project Structure

```
research-intelligence-engine/
├── app.py                      # Streamlit UI application
├── configs/
│   └── default.yaml            # Pipeline configuration
├── scripts/
│   ├── ingest.py               # Data ingestion script
│   └── evaluate.py             # Evaluation benchmark runner
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── pipeline.py             # RAG pipeline orchestrator
│   ├── data/
│   │   ├── arxiv_fetcher.py    # arXiv API client
│   │   └── models.py           # Data models (Paper, Chunk, Response)
│   ├── vectorstore/
│   │   ├── chunker.py          # Document chunking
│   │   ├── embedder.py         # Embedding generation
│   │   └── faiss_store.py      # FAISS index management
│   ├── retrieval/
│   │   └── retriever.py        # Retrieval pipeline
│   ├── generation/
│   │   └── generator.py        # Claude-based answer generation
│   └── evaluation/
│       ├── metrics.py          # Heuristic evaluation metrics
│       └── ragas_eval.py       # RAGAS integration
├── tests/
│   ├── test_arxiv_fetcher.py
│   ├── test_chunker.py
│   ├── test_config.py
│   ├── test_faiss_store.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_retriever.py
├── data/
│   ├── raw/                    # Fetched paper metadata (gitignored)
│   ├── processed/              # Evaluation results (gitignored)
│   └── vectordb/               # FAISS index files (gitignored)
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/rugwxd/research-intelligence-engine.git
cd research-intelligence-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Build the Index

```bash
# Fetch papers and build vector index (~15-20 min due to arXiv rate limits)
python scripts/ingest.py --papers 500
```

### Run the Application

```bash
# Start Streamlit UI
streamlit run app.py
```

### Run Evaluation

```bash
# Run benchmark evaluation
python scripts/evaluate.py

# Include RAGAS metrics (slower, requires more API calls)
python scripts/evaluate.py --ragas
```

## Usage Examples

### CLI Usage

```python
from src.pipeline import create_pipeline

pipeline = create_pipeline()
pipeline.load_index()

response = pipeline.query("What are the key innovations in transformer architectures?")

print(response.answer)
print(f"Sources: {len(response.sources)}")
print(f"Faithfulness: {response.eval_scores['faithfulness']:.3f}")
```

### Streamlit UI

The web interface provides:
- Chat-style query input with conversation history
- Expandable source cards with paper metadata and PDF links
- Real-time evaluation score badges (color-coded by quality)
- Configurable retrieval parameters via sidebar

## Configuration

All pipeline parameters are configurable via `configs/default.yaml`:

```yaml
arxiv:
  max_results_per_query: 150    # Papers per search topic
  rate_limit_seconds: 3.0       # arXiv API rate limit

chunking:
  chunk_size: 512               # Characters per chunk
  chunk_overlap: 64             # Overlap between chunks

retrieval:
  top_k: 10                     # Initial candidates from FAISS
  rerank_top_k: 5               # Final results after reranking
  similarity_threshold: 0.3     # Minimum cosine similarity

generation:
  model: "claude-sonnet-4-20250514"
  max_tokens: 2048
  temperature: 0.1
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_retriever.py -v
```

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **FAISS (Flat IP)** | Exact search optimal for <10K vectors; no approximate search overhead |
| **all-MiniLM-L6-v2** | Best balance of quality and speed for academic text at 384 dimensions |
| **Sliding window chunking** | Simple, effective for abstracts; overlap prevents information loss at boundaries |
| **Paper-level deduplication** | Prevents single papers from dominating results; improves answer diversity |
| **Dual evaluation** | Heuristic metrics for real-time UX; RAGAS for rigorous offline benchmarking |
| **Claude for generation** | Strong instruction following, citation compliance, and factual grounding |

## License

MIT
