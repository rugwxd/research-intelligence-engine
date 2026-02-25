"""Streamlit UI for the Research Intelligence Engine.

Provides a chat interface for querying AI/ML research papers
with real-time evaluation scores, source attribution, and
configurable retrieval modes.
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, setup_logging
from src.monitoring import health_check
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Research Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .source-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high { background-color: #d4edda; color: #155724; }
    .score-mid { background-color: #fff3cd; color: #856404; }
    .score-low { background-color: #f8d7da; color: #721c24; }
    .mode-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def get_score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-mid"
    return "score-low"


@st.cache_resource
def load_pipeline(retrieval_mode: str) -> RAGPipeline:
    """Initialize and cache the RAG pipeline."""
    config = load_config()
    setup_logging(config.logging)
    pipeline = RAGPipeline(config, retrieval_mode=retrieval_mode)

    try:
        pipeline.load_index()
    except FileNotFoundError:
        st.error(
            "Vector index not found. Run `python scripts/ingest.py` first "
            "to build the index."
        )
        st.stop()

    return pipeline


def render_sidebar():
    """Render the sidebar with configuration and info."""
    st.sidebar.title("Research Intelligence Engine")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Retrieval Mode")
    retrieval_mode = st.sidebar.radio(
        "Select retrieval strategy",
        options=["dense", "hybrid", "full"],
        index=1,
        help=(
            "**dense**: FAISS semantic search only\n\n"
            "**hybrid**: BM25 + FAISS with rank fusion\n\n"
            "**full**: hybrid + cross-encoder reranking"
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    top_k = st.sidebar.slider("Sources to retrieve", 3, 10, 5)
    use_ragas = st.sidebar.checkbox("Run RAGAS evaluation", value=False)
    use_llm_judge = st.sidebar.checkbox("Run LLM-as-Judge evaluation", value=False)

    st.sidebar.markdown("---")

    # Health check
    with st.sidebar.expander("System Health"):
        status = health_check()
        for component, info in status["components"].items():
            icon = "✅" if info["status"] in ("ready", "configured") else "⚠️"
            st.sidebar.text(f"{icon} {component}: {info['status']}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Example Queries")
    examples = [
        "What are the latest advances in transformer architectures?",
        "How does reinforcement learning from human feedback work?",
        "Compare different approaches to vision transformers",
        "What methods reduce hallucinations in large language models?",
        "Explain attention mechanisms and their variants",
    ]
    for example in examples:
        if st.sidebar.button(example, key=f"ex_{hash(example)}"):
            st.session_state["query_input"] = example

    return retrieval_mode, top_k, use_ragas, use_llm_judge


def render_scores(scores: dict[str, float]):
    """Render evaluation scores as colored badges."""
    # Filter to main scores for display
    display_scores = {
        k: v for k, v in scores.items()
        if not k.startswith("ragas_") and k != "overall"
    }
    if "overall" in scores:
        display_scores["overall"] = scores["overall"]

    cols = st.columns(min(len(display_scores), 4))
    for col, (metric, score) in zip(cols, display_scores.items()):
        css_class = get_score_class(score)
        col.markdown(
            f'<span class="score-badge {css_class}">'
            f"{metric.replace('_', ' ').title()}: {score:.2f}</span>",
            unsafe_allow_html=True,
        )

    # Show LLM judge and RAGAS scores separately if present
    extra_scores = {k: v for k, v in scores.items() if k.startswith(("ragas_", "llm_"))}
    if extra_scores:
        st.markdown("**Advanced Evaluation:**")
        cols2 = st.columns(min(len(extra_scores), 4))
        for col, (metric, score) in zip(cols2, extra_scores.items()):
            css_class = get_score_class(score)
            col.markdown(
                f'<span class="score-badge {css_class}">'
                f"{metric.replace('_', ' ').title()}: {score:.2f}</span>",
                unsafe_allow_html=True,
            )


def render_sources(sources):
    """Render source papers in expandable cards."""
    st.subheader("Source Papers")

    for i, result in enumerate(sources, 1):
        meta = result.chunk.metadata
        title = meta.get("title", "Unknown")
        authors = meta.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        arxiv_id = meta.get("arxiv_id", "")
        pdf_url = meta.get("pdf_url", "")

        with st.expander(f"[{i}] {title} (score: {result.score:.3f})"):
            st.markdown(f"**Authors:** {author_str}")
            st.markdown(f"**arXiv ID:** {arxiv_id}")
            if pdf_url:
                st.markdown(f"[View PDF]({pdf_url})")
            st.markdown("---")
            st.markdown(f"**Retrieved passage:**\n\n{result.chunk.text}")


def main():
    """Main Streamlit application."""
    retrieval_mode, top_k, use_ragas, use_llm_judge = render_sidebar()

    st.title("Research Intelligence Engine")
    st.caption(
        "Semantic search over 500+ AI/ML research papers with "
        "hybrid retrieval, cross-encoder reranking, and multi-layer evaluation"
    )

    # Show retrieval mode indicator
    mode_colors = {"dense": "#3b82f6", "hybrid": "#8b5cf6", "full": "#059669"}
    st.markdown(
        f'<span class="mode-tag" style="background-color: {mode_colors[retrieval_mode]}; '
        f'color: white;">Mode: {retrieval_mode.upper()}</span>',
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query input
    query = st.chat_input("Ask a question about AI/ML research...")

    # Handle example queries from sidebar
    if "query_input" in st.session_state and st.session_state["query_input"]:
        query = st.session_state.pop("query_input")

    if query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                pipeline = load_pipeline(retrieval_mode)

                # Update retrieval config
                pipeline.retriever.config.rerank_top_k = top_k

                response = pipeline.query(
                    query,
                    use_ragas=use_ragas,
                    use_llm_judge=use_llm_judge,
                )

            # Display answer
            st.markdown(response.answer)

            # Display metrics
            st.markdown("---")
            st.subheader("Evaluation Scores")
            render_scores(response.eval_scores)

            # Display latency
            st.caption(f"Response time: {response.latency_ms:.0f}ms | Mode: {retrieval_mode}")

            # Display sources
            render_sources(response.sources)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
        })


if __name__ == "__main__":
    main()
