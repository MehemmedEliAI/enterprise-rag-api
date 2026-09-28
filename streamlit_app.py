import os
import time
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Enterprise Multimodal RAG Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #475569; margin-bottom: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 6px; height: 2.8em; }
    .metric-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Sidebar Controls
with st.sidebar:
    st.title("⚡ Enterprise RAG")
    st.caption("Production Stack: Qdrant | Redis | Groq")
    st.divider()

    # System Status Indicator
    st.subheader("System Status")
    try:
        health_res = requests.get(f"{API_URL}/docs", timeout=2)
        if health_res.status_code == 200:
            st.success("API Backend: Online")
        else:
            st.warning("API Backend: Degraded")
    except Exception:
        st.error("API Backend: Offline")

    st.divider()

    # Document Upload Section
    st.subheader("📁 Ingest Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload files to embed and store in Qdrant Vector Store"
    )

    if uploaded_file is not None:
        if st.button("Ingest File", type="primary"):
            with st.spinner("Processing & Indexing Document..."):
                try:
                    start_time = time.time()
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/api/v1/ingest", files=files, timeout=60)
                    duration = round(time.time() - start_time, 2)

                    if response.status_code == 200:
                        st.success(f"Successfully Ingested in {duration}s!")
                        if uploaded_file.name not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(uploaded_file.name)
                    else:
                        st.error(f"Ingestion Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

    st.divider()

    # Search Configuration
    st.subheader("⚙️ Query Settings")
    top_k = st.slider("Context Chunks (top_k)", min_value=1, max_value=10, value=3)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Interface Layout
st.markdown('<div class="main-header">Enterprise Multimodal RAG Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Vector Search with Re-ranking, Redis Caching, and Async Groq LLM</div>', unsafe_allow_html=True)

# Tab Navigation
tab_chat, tab_docs, tab_system = st.tabs(["💬 Query & Chat", "📚 Document Library", "ℹ️ System Architecture"])

# Tab 1: Chat Interface
with tab_chat:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "contexts" in msg and msg["contexts"]:
                with st.expander("📄 View Retrieved Context Chunks"):
                    for idx, ctx in enumerate(msg["contexts"], 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.text(ctx)

    # User Input Field
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector store and generating response..."):
                start_time = time.time()
                try:
                    payload = {"query": prompt, "top_k": top_k}
                    response = requests.post(f"{API_URL}/api/v1/query", json=payload, timeout=30)
                    latency = round(time.time() - start_time, 2)

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No response generated.")
                        contexts = data.get("contexts", [])
                        is_cached = data.get("cached", False)

                        st.write(answer)

                        # Response Metadata Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"⏱️ **Latency:** {latency}s")
                        with col2:
                            cache_label = "⚡ Redis Cache Hit" if is_cached else "🔍 Vector DB Search"
                            st.caption(f"💾 **Source:** {cache_label}")
                        with col3:
                            st.caption(f"📚 **Retrieved Chunks:** {len(contexts)}")

                        # Display Context Accordion
                        if contexts:
                            with st.expander("📄 View Retrieved Context Chunks"):
                                for idx, ctx in enumerate(contexts, 1):
                                    st.markdown(f"**Context Chunk {idx}:**")
                                    st.code(ctx, language="text")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "contexts": contexts
                        })
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Request Error: {e}")

# Tab 2: Document Library
with tab_docs:
    st.subheader("Indexed Documents")
    if st.session_state.ingested_files:
        for file in st.session_state.ingested_files:
            st.success(f"📄 **{file}** - Active in Qdrant Vector Collection")
    else:
        st.info("No documents ingested in this session yet. Use the sidebar to upload files.")

# Tab 3: System Architecture
with tab_system:
    st.subheader("System Stack Details")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Vector Database", value="Qdrant")
        st.caption("Embedding Model: BAAI/bge-small-en-v1.5")
    with col_b:
        st.metric(label="Re-ranker Model", value="FlashRank")
        st.caption("Model: ms-marco-TinyBERT-L-2-v2")
    with col_c:
        st.metric(label="LLM Provider", value="Groq API")
        st.caption("Model: openai/gpt-oss-20b")