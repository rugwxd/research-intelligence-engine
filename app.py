"""Streamlit UI for the Research Intelligence Engine.

Provides a chat interface for querying AI/ML research papers
with real-time evaluation scores and source attribution.
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, setup_logging
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
</style>
""", unsafe_allow_html=True)


def get_score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-mid"
    return "score-low"


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """Initialize and cache the RAG pipeline."""
    config = load_config()
    setup_logging(config.logging)
    pipeline = RAGPipeline(config)

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

    st.sidebar.subheader("Settings")
    top_k = st.sidebar.slider("Sources to retrieve", 3, 10, 5)
    use_ragas = st.sidebar.checkbox("Run RAGAS evaluation", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        "Query 500+ AI/ML research papers from arXiv using "
        "semantic search and grounded answer generation."
    )

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

    return top_k, use_ragas


def render_scores(scores: dict[str, float]):
    """Render evaluation scores as colored badges."""
    cols = st.columns(len(scores))
    for col, (metric, score) in zip(cols, scores.items()):
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
    top_k, use_ragas = render_sidebar()

    st.title("Research Intelligence Engine")
    st.caption("Semantic search over 500+ AI/ML research papers with grounded answers")

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
                pipeline = load_pipeline()

                # Update retrieval config
                pipeline.retriever.config.rerank_top_k = top_k

                response = pipeline.query(query, use_ragas=use_ragas)

            # Display answer
            st.markdown(response.answer)

            # Display metrics
            st.markdown("---")
            st.subheader("Evaluation Scores")
            render_scores(response.eval_scores)

            # Display latency
            st.caption(f"Response time: {response.latency_ms:.0f}ms")

            # Display sources
            render_sources(response.sources)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
        })


if __name__ == "__main__":
    main()
