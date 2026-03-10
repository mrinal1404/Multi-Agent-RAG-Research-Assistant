"""
Streamlit UI: query history, confidence scores, citation traceability,
streaming responses, and document upload.
"""

import json
import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────── Session State ────────────────

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ──────────────── Sidebar ────────────────

with st.sidebar:
    st.title("🔬 RAG Research Assistant")
    st.caption("Multi-Agent | FAISS + BM25 | LangGraph")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        stats = requests.get(f"{API_BASE}/stats", timeout=3).json()
        if health["index_ready"]:
            st.success("✅ Index Ready")
            st.metric("Indexed Chunks", stats.get("total_indexed_chunks", 0))
        else:
            st.warning("⚠️ Index Building...")
    except Exception:
        st.error("❌ API Offline — start the FastAPI server")

    st.divider()
    st.subheader("📤 Upload Paper")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded and st.button("Upload & Index"):
        with st.spinner("Uploading..."):
            resp = requests.post(
                f"{API_BASE}/index/upload",
                files={"file": (uploaded.name, uploaded.getvalue())}
            )
            if resp.status_code == 200:
                st.success("Uploaded! Click Rebuild Index.")
            else:
                st.error(resp.text)

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding..."):
            requests.post(f"{API_BASE}/index/rebuild")
            st.success("Rebuild started in background.")

    st.divider()
    st.subheader("📋 Query History")
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(f"🔍 {item['query'][:35]}...", key=f"hist_{i}"):
                st.session_state.current_result = item
    else:
        st.caption("No queries yet.")

    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.session_state.current_result = None
        st.rerun()

# ──────────────── Main UI ────────────────

st.title("🔬 Multi-Agent RAG Research Assistant")
st.caption("Powered by LangGraph · FAISS Dense + BM25 Sparse · Reciprocal Rank Fusion")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "Enter your research query",
        placeholder="e.g. How does BERT use masked language modeling for pre-training?",
        height=100
    )
with col2:
    st.write("")
    st.write("")
    use_streaming = st.checkbox("⚡ Stream Response", value=True)
    run_query = st.button("🚀 Search", type="primary", use_container_width=True)

st.divider()

# ──────────────── Query Execution ────────────────

if run_query and query.strip():
    result_container = st.container()

    with result_container:
        if use_streaming:
            st.subheader("📡 Streaming Response")
            answer_box = st.empty()
            status_box = st.empty()
            token_buffer = []

            try:
                with requests.post(
                    f"{API_BASE}/query/stream",
                    json={"query": query, "stream": True},
                    stream=True,
                    timeout=60
                ) as resp:
                    citations_data = []
                    confidence = 0.0

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if data["type"] == "status":
                                    status_box.info(f"🔄 {data['message']}")
                                elif data["type"] == "retrieval":
                                    status_box.success(f"📚 Retrieved {data['count']} documents")
                                elif data["type"] == "token":
                                    token_buffer.append(data["content"])
                                    answer_box.markdown("".join(token_buffer))
                                elif data["type"] == "complete":
                                    citations_data = data.get("citations", [])
                                    confidence = data.get("confidence_score", 0)
                                    status_box.empty()
                            except json.JSONDecodeError:
                                pass

                final_answer = "".join(token_buffer)
                st.session_state.query_history.append({
                    "query": query,
                    "final_answer": final_answer,
                    "confidence_score": confidence,
                    "citations": citations_data,
                    "reasoning_trace": []
                })

            except Exception as e:
                st.error(f"Streaming error: {e}. Falling back to standard query.")
                use_streaming = False

        if not use_streaming:
            with st.spinner("🤔 Running multi-agent pipeline..."):
                start = time.time()
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"query": query},
                        timeout=60
                    )
                    elapsed = time.time() - start

                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.current_result = result
                        st.session_state.query_history.append(result)

                        st.success(f"✅ Completed in {elapsed:.2f}s")

                        # Display answer
                        st.subheader("📝 Answer")
                        st.markdown(result["final_answer"])
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ──────────────── Display Current Result ────────────────

result_to_show = st.session_state.current_result
if result_to_show and not run_query:
    st.subheader("📝 Answer")
    st.markdown(result_to_show.get("final_answer", ""))

if result_to_show or (run_query and not use_streaming):
    # Pick result
    r = st.session_state.current_result or {}
    if not r and st.session_state.query_history:
        r = st.session_state.query_history[-1]

    if r:
        col_conf, col_docs = st.columns(2)

        with col_conf:
            conf = r.get("confidence_score", 0)
            color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
            st.metric(
                f"{color} Confidence Score",
                f"{conf:.0%}",
                help="Based on claim verification against source documents"
            )

        with col_docs:
            st.metric(
                "📚 Documents Retrieved",
                r.get("retrieved_docs_count", len(r.get("citations", [])))
            )

        # Citations
        citations = r.get("citations", [])
        if citations:
            st.subheader("📎 Citations & Sources")
            for i, cite in enumerate(citations, 1):
                with st.expander(f"[{i}] {cite.get('source', 'Unknown')}"):
                    st.write(cite.get("relevant_quote", "No excerpt available."))

        # Reasoning trace
        trace = r.get("reasoning_trace", [])
        if trace:
            st.subheader("🧠 Agent Reasoning Trace")
            with st.expander("View full trace", expanded=False):
                for step in trace:
                    icon = "🔍" if "Retrieval" in step else "✍️" if "Summar" in step else "✅" if "Fact" in step else "⚙️"
                    st.markdown(f"{icon} `{step}`")

# ──────────────── Footer ────────────────

st.divider()
st.caption("Multi-Agent RAG Pipeline: RetrievalAgent → SummarizationAgent → FactCheckingAgent | Hybrid FAISS + BM25 + RRF")
