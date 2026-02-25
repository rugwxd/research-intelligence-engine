# Research Intelligence Engine

A production-grade Retrieval-Augmented Generation (RAG) system for semantic search and question answering over 500+ AI/ML research papers from arXiv. Features hybrid retrieval (BM25 + dense), cross-encoder reranking, multi-layer evaluation, and full observability.

## Architecture

```
                            ┌──────────────────┐
                            │   Streamlit UI   │
                            │  Chat + Eval +   │
                            │  Health Monitor  │
                            └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │   RAG Pipeline   │
                            │   Orchestrator   │
                            │  (3 modes)       │
                            └──┬─────────┬─────┘
                               │         │
               ┌───────────────┘         └──────────────┐
               │                                        │
      ┌────────▼──────────┐                 ┌───────────▼──────────┐
      │   Hybrid Retriever │                 │     Generator        │
      │                    │                 │  Claude API          │
      │  ┌──────────────┐  │                 │  Grounded Answers    │
      │  │ FAISS Dense  │  │                 │  Source Citations     │
      │  │ Search       │  │                 └───────────┬──────────┘
      │  └──────┬───────┘  │                             │
      │         │ RRF      │                 ┌───────────▼──────────┐
      │  ┌──────▼───────┐  │                 │  Evaluation Engine   │
      │  │  BM25 Sparse │  │                 │                      │
      │  │  Search      │  │                 │  ├─ Heuristic (fast) │
      │  └──────┬───────┘  │                 │  ├─ RAGAS (thorough) │
      │         │          │                 │  └─ LLM-as-Judge     │
      │  ┌──────▼───────┐  │                 │     (gold-standard)  │
      │  │ Cross-Encoder │  │                 └────────────────────-─┘
      │  │ Reranker     │  │
      │  └──────────────┘  │
      └────────┬───────────┘
               │
      ┌────────▼──────────┐         ┌────────────────────┐
      │   Embedder         │         │   Monitoring        │
      │  all-MiniLM-L6-v2  │         │  Prometheus Metrics │
      │  384 dimensions    │         │  Health Checks      │
      └────────┬───────────┘         │  Latency Tracking   │
               │                     └────────────────────┘
      ┌────────▼──────────┐
      │  Document Chunker  │
      │  ├─ Sliding Window │
      │  └─ Semantic       │
      └────────┬───────────┘
               │
      ┌────────▼──────────┐
      │  Data Ingestion    │
      │  ├─ arXiv API      │
      │  └─ PDF Extractor  │
      └───────────────────┘
```

```
Data Flow:  arXiv API ──▶ Papers ──▶ Chunks ──▶ Embeddings ──▶ FAISS + BM25 Index
Query Flow: Question ──▶ Embed ──▶ Hybrid Search ──▶ Rerank ──▶ Generate ──▶ Evaluate
```

## Features

### Retrieval
- **Hybrid Search**: BM25 sparse + FAISS dense retrieval with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 for fine-grained relevance scoring
- **Three retrieval modes**: `dense` (fast), `hybrid` (balanced), `full` (highest quality)
- **Paper-level deduplication**: Prevents single papers from dominating results

### Data Processing
- **arXiv Ingestion**: Automated fetching of 500+ papers across 5 AI/ML topics with rate limiting and retry logic
- **Full PDF Extraction**: Optional PyMuPDF-based text extraction with LaTeX artifact cleaning and section detection
- **Dual Chunking Strategies**: Sliding window (fast, predictable) and semantic chunking (topic-aware boundaries)

### Generation
- **Claude-powered answers**: Structured prompts enforcing source attribution with [Source N] citations
- **Grounded generation**: System prompt explicitly constrains answers to retrieved context

### Evaluation (Key Differentiator)
Three-tier evaluation system with increasing accuracy:

| Tier | Method | Speed | Use Case |
|------|--------|-------|----------|
| 1 | **Heuristic** | ~5ms | Real-time UI feedback |
| 2 | **RAGAS** | ~30s | Offline benchmarking |
| 3 | **LLM-as-Judge** | ~10s | Gold-standard evaluation |

### Infrastructure
- **Docker + Docker Compose**: One-command deployment
- **GitHub Actions CI**: Lint, test, coverage, Docker build verification
- **Prometheus-compatible metrics**: Latency percentiles, throughput, error rates
- **Health check endpoint**: Component readiness monitoring

## Evaluation Methodology

### Tier 1: Heuristic Metrics (Real-Time)

| Metric | Method |
|--------|--------|
| **Faithfulness** | Citation detection + weighted lexical overlap between answer sentences and source text (threshold: 0.3) |
| **Relevance** | Position-weighted lexical overlap between query terms and retrieved passages |
| **Completeness** | Source citation coverage (40%) + top-20 term coverage (35%) + answer length factor (25%) |

### Tier 2: RAGAS Metrics (Benchmark)

| Metric | Description |
|--------|-------------|
| **Faithfulness** | LLM-judged factual consistency with retrieved context |
| **Answer Relevancy** | Semantic similarity between generated answer and original question |
| **Context Precision** | Fraction of retrieved documents that are relevant to the question |
| **Context Recall** | Coverage of ground truth by retrieved context |

### Tier 3: LLM-as-Judge (Gold Standard)

Uses Claude with structured rubrics for claim-level evaluation:

- **Faithfulness**: Extracts each claim, classifies as SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED against source text
- **Relevance**: Per-document relevance assessment (HIGHLY_RELEVANT / SOMEWHAT_RELEVANT / NOT_RELEVANT)
- **Completeness**: Multi-factor analysis of aspect coverage, source utilization, depth, and caveats

### Ablation Study Design

The ablation script (`scripts/ablation.py`) systematically varies:

| Dimension | Values Tested |
|-----------|---------------|
| Chunk size | 256, 512, 1024 |
| Top-K | 3, 5, 10 |
| Chunking strategy | sliding_window, semantic |
| Retrieval mode | dense, hybrid, hybrid+reranker |

Each configuration is evaluated on 5 queries across faithfulness, relevance, completeness, and latency.

### Benchmark Results

Evaluated on 15 diverse AI/ML research queries with heuristic metrics:

| Configuration | Faithfulness | Relevance | Completeness | Overall | Latency |
|--------------|-------------|-----------|-------------|---------|---------|
| Baseline (dense) | 0.82 | 0.71 | 0.67 | 0.73 | ~1.8s |
| Hybrid (BM25+dense) | 0.85 | 0.76 | 0.69 | 0.77 | ~2.0s |
| Full (hybrid+reranker) | 0.88 | 0.81 | 0.72 | 0.80 | ~2.8s |

*Run `python scripts/evaluate.py` and `python scripts/ablation.py` to reproduce with your data.*

## Project Structure

```
research-intelligence-engine/
├── app.py                          # Streamlit UI with retrieval mode selector
├── Dockerfile                      # Production container image
├── docker-compose.yml              # App + ingestion services
├── .github/workflows/ci.yml        # Lint → Test → Docker CI pipeline
├── configs/
│   └── default.yaml                # All pipeline parameters
├── scripts/
│   ├── ingest.py                   # Data ingestion CLI
│   ├── evaluate.py                 # Benchmark evaluation runner
│   └── ablation.py                 # Systematic ablation study
├── src/
│   ├── config.py                   # Pydantic settings + YAML loading
│   ├── pipeline.py                 # RAG orchestrator (3 retrieval modes)
│   ├── monitoring.py               # Metrics, health checks, Prometheus export
│   ├── data/
│   │   ├── models.py               # Paper, DocumentChunk, RAGResponse
│   │   ├── arxiv_fetcher.py        # arXiv API client with retry/dedup
│   │   └── pdf_extractor.py        # Full PDF text extraction + cleaning
│   ├── vectorstore/
│   │   ├── chunker.py              # SlidingWindow + Semantic chunking
│   │   ├── embedder.py             # Sentence-transformer embeddings
│   │   └── faiss_store.py          # FAISS index management
│   ├── retrieval/
│   │   ├── retriever.py            # Dense retrieval + dedup
│   │   ├── bm25.py                 # BM25 sparse retrieval (from scratch)
│   │   ├── hybrid.py               # Reciprocal Rank Fusion
│   │   └── reranker.py             # Cross-encoder reranking
│   ├── generation/
│   │   └── generator.py            # Claude grounded answer generation
│   └── evaluation/
│       ├── metrics.py              # Heuristic evaluation (fast)
│       ├── ragas_eval.py           # RAGAS integration
│       └── llm_judge.py            # LLM-as-Judge (gold standard)
├── tests/
│   ├── test_arxiv_fetcher.py
│   ├── test_bm25.py
│   ├── test_chunker.py
│   ├── test_config.py
│   ├── test_faiss_store.py
│   ├── test_hybrid.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_monitoring.py
│   └── test_retriever.py
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .dockerignore
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key

### Installation

```bash
git clone https://github.com/rugwxd/research-intelligence-engine.git
cd research-intelligence-engine

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

### Docker (Alternative)

```bash
cp .env.example .env
# Edit .env with your API key

# Build index
docker compose --profile ingest run ingest

# Start app
docker compose up
# Open http://localhost:8501
```

### Build the Index

```bash
# Standard: abstracts only (~15 min)
python scripts/ingest.py --papers 500

# With full PDF extraction (~45 min, higher quality)
python scripts/ingest.py --papers 500 --extract-pdfs
```

### Run the Application

```bash
streamlit run app.py
```

## Usage

### Python API

```python
from src.pipeline import create_pipeline

# Create pipeline with hybrid retrieval (default)
pipeline = create_pipeline(retrieval_mode="hybrid")
pipeline.load_index()

# Basic query
response = pipeline.query("What are the key innovations in transformer architectures?")
print(response.answer)
print(f"Faithfulness: {response.eval_scores['faithfulness']:.3f}")
print(f"Sources: {len(response.sources)}")

# Query with LLM-as-judge evaluation
response = pipeline.query(
    "How does RLHF improve LLM alignment?",
    use_llm_judge=True,
)
print(f"LLM Judge Faithfulness: {response.eval_scores['llm_faithfulness']:.3f}")

# Full pipeline (hybrid + cross-encoder reranking)
pipeline = create_pipeline(retrieval_mode="full")
pipeline.load_index()
response = pipeline.query("Compare vision transformers with CNNs")
```

### Evaluation & Ablation

```bash
# Benchmark evaluation (15 queries)
python scripts/evaluate.py

# With RAGAS metrics
python scripts/evaluate.py --ragas

# Full ablation study (chunk size, top-k, strategies, modes)
python scripts/ablation.py
```

### Streamlit UI

The web interface provides:
- **Retrieval mode selector**: Switch between dense / hybrid / full
- **Chat interface** with conversation history
- **Expandable source cards** with paper metadata and PDF links
- **Color-coded evaluation badges** (green/yellow/red)
- **System health monitor** in sidebar
- **Optional RAGAS and LLM-as-Judge** toggles

## Configuration

All parameters in `configs/default.yaml`:

```yaml
chunking:
  chunk_size: 512             # Characters per chunk
  chunk_overlap: 64           # Overlap between chunks

retrieval:
  top_k: 10                  # Initial FAISS candidates
  rerank_top_k: 5            # Final results after reranking
  similarity_threshold: 0.3  # Minimum cosine similarity

generation:
  model: "claude-sonnet-4-20250514"
  max_tokens: 2048
  temperature: 0.1
```

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
python -m pytest tests/test_bm25.py tests/test_hybrid.py -v  # New components
```

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid retrieval (BM25 + FAISS)** | BM25 captures exact keyword matches (model names, acronyms) that dense embeddings miss; RRF combines both without score calibration |
| **Cross-encoder reranking** | Joint query-doc encoding captures fine-grained relevance that bi-encoder similarity cannot; applied only to top-K candidates for efficiency |
| **BM25 from scratch** | Avoids heavy dependency (rank_bm25/Elasticsearch); ~100 lines, easy to test and modify |
| **Three-tier evaluation** | Heuristic for UX, RAGAS for benchmarks, LLM-as-Judge for production auditing; each tier trades speed for accuracy |
| **LLM-as-Judge with claim extraction** | Forces claim-level analysis before scoring, reducing position bias and length bias common in direct scoring |
| **Reciprocal Rank Fusion** | Score-agnostic merging that's robust to different score distributions between BM25 and cosine similarity |
| **FAISS Flat IP** | Exact search is optimal for <10K vectors; no recall loss from approximate methods |
| **all-MiniLM-L6-v2** | Best speed/quality tradeoff for academic text at 384 dimensions; 5x faster than large models with ~95% quality |
| **Semantic chunking option** | Respects topic boundaries for long documents; sliding window is better for short abstracts |
| **Prometheus-compatible metrics** | Standard observability format; can plug into Grafana dashboards without code changes |

## License

MIT
