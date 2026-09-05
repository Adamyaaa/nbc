import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

# Load env before importing local modules
load_dotenv()

from app.core.llm import LLMClient
from app.retrieval.vector_store import VectorStore
from app.retrieval.engine import RetrievalEngine
from app.services.rag import RAGService
from app.evaluation.judge import LLMJudge
from app.evaluation.runner import ExperimentRunner

st.set_page_config(page_title="AI Evaluation Platform", layout="wide")

# --- Dependency Initialization (Cached) ---
@st.cache_resource
def get_services():
    llm_client = LLMClient()
    vector_store = VectorStore()
    retrieval_engine = RetrievalEngine(vector_store, llm_client)
    rag_service = RAGService(llm_client, retrieval_engine)
    judge = LLMJudge(llm_client)
    runner = ExperimentRunner(rag_service, judge)
    return rag_service, runner

try:
    rag_service, runner = get_services()
except Exception as e:
    st.error(f"Failed to initialize services. Is GEMINI_API_KEY set in .env? Error: {e}")
    st.stop()

# --- UI Sidebar ---
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("Adjust the retrieval parameters for your query.")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)
    model = st.selectbox("LLM Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.divider()
    st.markdown("**Platform Status:** `Online`")
    st.markdown(f"**Vector Store Size:** `{rag_service.retrieval_engine.vector_store.index.ntotal}` chunks")

# --- Main Layout ---
st.title("Production RAG & Evaluation Platform")

tab1, tab2 = st.tabs(["💬 Chat with Data", "📊 Evaluation Dashboard"])

# --- TAB 1: CHAT ---
with tab1:
    st.markdown("### Ask Questions against your FAISS Vector Store")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    answer = rag_service.answer_question(query=prompt, top_k=top_k, model=model)
                    
                    response_text = f"{answer.answer}\n\n"
                    response_text += f"---\n*Confidence:* `{answer.confidence}` | *Latency:* `{answer.metadata.get('latency_seconds', 0):.2f}s` | *Sources:* `{len(answer.sources)} chunks`"
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"An error occurred: {e}")


# --- TAB 2: EVALUATION DASHBOARD ---
with tab2:
    st.markdown("### LLM-as-a-Judge Experiment Runner")
    st.markdown("Run a benchmark dataset through the RAG pipeline and automatically grade the results on **Correctness**, **Relevance**, **Groundedness**, **Context Precision**, and **Context Recall**.")
    
    # Dummy dataset for UI demo
    default_dataset = [
        {"question": "Where is Acme Corp located?", "expected_answer": "Austin, Texas"},
        {"question": "When was Acme Corp founded?", "expected_answer": "2015"},
    ]
    
    st.write("Current Evaluation Dataset (Preview):")
    st.json(default_dataset)
    
    if st.button("🚀 Run Benchmark Evaluation"):
        with st.spinner("Running experiments and judging answers... This may take a minute..."):
            try:
                report = runner.run_experiment(default_dataset, top_k=top_k, model=model)
                st.success("Evaluation Complete!")
                
                # Display Summary Metrics
                summary = report["summary"]
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Correctness", f"{summary['avg_correctness']:.2f}")
                col2.metric("Relevance", f"{summary['avg_relevance']:.2f}")
                col3.metric("Groundedness", f"{summary['avg_groundedness']:.2f}")
                col4.metric("Ctx Precision", f"{summary['avg_context_precision']:.2f}")
                col5.metric("Ctx Recall", f"{summary['avg_context_recall']:.2f}")
                
                # Detailed results table
                st.markdown("#### Detailed Results")
                df_data = []
                for detail in report["details"]:
                    eval_scores = detail["evaluation"]
                    df_data.append({
                        "Question": detail["question"],
                        "Generated Answer": detail["generated"],
                        "Correctness": eval_scores["correctness"],
                        "Groundedness": eval_scores["groundedness"],
                        "Latency (s)": round(detail["latency"], 2)
                    })
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
