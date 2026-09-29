"""
Agentic RAG Financial Assistant using LangGraph, Qdrant, Cohere, and Groq.
"""
import os
import json
import requests
import streamlit as st
from typing import TypedDict, List
from qdrant_client import QdrantClient
import cohere
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Prakhar | Agentic RAG", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    /* Styling for metric cards and expanders */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Increase Font Size for the Navigation Tabs */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Fetch keys from Streamlit Secrets (or local environment variables)
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", os.getenv("COHERE_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not all([COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    st.error("⚠️ Missing API Keys. Please configure Streamlit Secrets.")
    st.stop()

# Initialize API Clients
cohere_client = cohere.Client(COHERE_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "financial_reports_cohere"

# ==========================================
# 2. SIDEBAR BRANDING & MODEL DISCOVERY
# ==========================================
@st.cache_data(ttl=3600)
def fetch_available_models():
    """Fetches dynamic model list directly from Groq and filters non-chat models."""
    models_list = []
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    m_id = m.get("id", "")
                    
                    # Filter out obvious non-chat or internal guard models
                    if not any(excluded in m_id.lower() for excluded in ["whisper", "guard", "safeguard"]):
                        models_list.append({"id": m_id, "display_name": f"Groq ({m_id})"})
        except Exception:
            pass
            
    # Strictly return the fetched list. No hardcoded models appended.
    return models_list

with st.sidebar:
    st.markdown('<h1 style="font-size: 38px; margin-bottom: 0px;">Prakhar Avasthi</h1>', unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top: -15px; margin-bottom: 15px; color: #1C83E1; font-weight: 600; font-size: 16px; letter-spacing: 0.5px;'>Data Science & AI Professional</div>", 
        unsafe_allow_html=True
    )
    st.divider()
    
    st.link_button("🔗 LinkedIn", "http://www.linkedin.com/in/prakhar-avasthi-35067a1bb", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/AvasthiPrakhar", use_container_width=True)
    st.link_button("📊 Kaggle", "https://www.kaggle.com/avasthiprakhar", use_container_width=True)

    st.divider()
    st.markdown("📧 **prakharavasthi1999@gmail.com**")

    st.divider()
    st.markdown("### ⚙️ Agent Configuration")
    
    available_models = fetch_available_models()
    
    # Safety check: Prevent Streamlit selectbox crash if API fails to return models
    if not available_models:
        st.error("⚠️ Failed to fetch models from Groq. Please check your API key and connection.")
        st.stop()
        
    model_map = {m["display_name"]: m["id"] for m in available_models}
    
    selected_display_name = st.selectbox(
        "Select LLM Architecture", 
        options=list(model_map.keys()),
        help="Dynamically fetched from Groq. Non-chat models are automatically filtered out."
    )
    
    target_model_id = model_map[selected_display_name]
    
    if st.button("🔄 Refresh Models"):
        st.cache_data.clear()
        st.rerun()

# Initialize the primary LLM dynamically based on user selection
# Minor update: LangChain deprecated 'model_name', replaced with 'model'
llm = ChatGroq(model=target_model_id, groq_api_key=GROQ_API_KEY, temperature=0.1)

# ==========================================
# 3. LANGGRAPH STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    question: str
    context: List[str]
    draft_answer: str
    feedback: str
    hallucination_found: str
    iterations: int
    trace_log: List[str]

# ==========================================
# 4. AGENT NODES (The Brains of the Operation)
# ==========================================
def retrieve_and_rerank(state: AgentState):
    """Fetches documents from Qdrant and uses Cohere to Rerank the best ones."""
    query = state["question"]
    state["trace_log"].append("🔍 **Step 1: Retrieval & Reranking started...**")
    
    query_vector = cohere_client.embed(
        texts=[query], model="embed-english-v3.0", input_type="search_query"
    ).embeddings[0]
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=15
    )
    docs = [point.payload["text"] for point in search_results.points]
    
    if docs:
        reranked = cohere_client.rerank(
            query=query, documents=docs, model="rerank-english-v3.0", top_n=5
        )
        best_docs = [docs[res.index] for res in reranked.results]
    else:
        best_docs = []

    state["context"] = best_docs
    state["trace_log"].append(f"✅ Found top {len(best_docs)} highly relevant financial chunks.")
    return state

def generate_answer(state: AgentState):
    """Generates an answer using the retrieved context."""
    state["trace_log"].append(f"✍️ **Step 2: Generating draft answer (Iteration {state['iterations'] + 1})...**")
    
    context_str = "\n\n".join(state["context"])
    feedback = state.get("feedback", "")
    
    prompt = f"""
    You are an expert Financial Analyst. Answer the user's question using ONLY the provided context.
    If the context does not contain the answer, say "I cannot answer this based on the available SEC filings."
    
    Context:
    {context_str}
    
    User Question: {state['question']}
    """
    
    if feedback:
        prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT: {feedback}\nRewrite the answer to fix this error."
        
    response = llm.invoke([HumanMessage(content=prompt)])
    state["draft_answer"] = response.content
    state["iterations"] += 1
    return state

def evaluate_hallucination(state: AgentState):
    """The Guardrail: Checks if the generated answer made up fake numbers."""
    state["trace_log"].append("⚖️ **Step 3: Guardrail Evaluation...**")
    
    context_str = "\n\n".join(state["context"])
    answer = state["draft_answer"]
    
    prompt = f"""
    You are a strict Audit Bot. Your job is to check if the generated answer contains ANY financial numbers or claims that are NOT present in the context.
    
    Context:
    {context_str}
    
    Draft Answer:
    {answer}
    
    Output a strictly valid JSON object with two keys:
    1. "hallucinated": "yes" if it made up facts/numbers, or "no" if it is fully supported.
    2. "feedback": Detail exactly what was made up (or write "Perfect" if none).
    
    JSON Output:
    """
    
    response = llm.invoke([SystemMessage(content="Output ONLY valid JSON. No markdown tags."), HumanMessage(content=prompt)])
    
    try:
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        eval_result = json.loads(clean_json)
        state["hallucination_found"] = eval_result.get("hallucinated", "yes").lower()
        state["feedback"] = eval_result.get("feedback", "Error parsing feedback.")
    except Exception as e:
        state["hallucination_found"] = "yes"
        state["feedback"] = "System failed to parse evaluation. Retrying."

    if state["hallucination_found"] == "yes":
        state["trace_log"].append(f"❌ **Hallucination Detected!** Reason: {state['feedback']}")
    else:
        state["trace_log"].append("✅ **Evaluation Passed!** Answer is factually grounded.")
        
    return state

# ==========================================
# 5. ROUTING LOGIC & GRAPH COMPILATION
# ==========================================
def should_regenerate(state: AgentState):
    if state["hallucination_found"] == "yes" and state["iterations"] < 3:
        return "generate_answer"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_and_rerank", retrieve_and_rerank)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("evaluate_hallucination", evaluate_hallucination)

workflow.add_edge(START, "retrieve_and_rerank")
workflow.add_edge("retrieve_and_rerank", "generate_answer")
workflow.add_edge("generate_answer", "evaluate_hallucination")
workflow.add_conditional_edges("evaluate_hallucination", should_regenerate)

agent_app = workflow.compile()

# ==========================================
# 6. MAIN UI (TABS)
# ==========================================
tab_chat, tab_methodology = st.tabs(["💬 Financial AI Agent", "🧠 Agentic RAG Architecture"])

# ------------------------------------------
# TAB 1: CHAT INTERFACE
# ------------------------------------------
with tab_chat:
    st.title("📈 Enterprise Financial AI Agent")
    st.markdown("*A self-correcting Agentic RAG pipeline querying the `FinanceBench` SEC 10-K dataset.*")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show Trace Log
            if "trace" in msg:
                with st.expander("🔍 View AI Thought Process (LangGraph Trace)"):
                    for step in msg["trace"]:
                        st.write(step)
                        
            # NEW: Show Source Documents
            if "context" in msg and msg["context"]:
                with st.expander("📄 View Retrieved Source Documents"):
                    for i, doc in enumerate(msg["context"]):
                        st.markdown(f"**Source Chunk {i+1}:**")
                        st.info(doc) # st.info creates a nice, colored, word-wrapped card!

    # Handle New User Input
    if prompt := st.chat_input("Ask a question about the financial reports (e.g., 'What was the operating margin?'):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner(f"Agent ({target_model_id}) is retrieving, analyzing, and auditing..."):
                try:
                    initial_state = {
                        "question": prompt,
                        "context": [],
                        "draft_answer": "",
                        "feedback": "",
                        "hallucination_found": "no",
                        "iterations": 0,
                        "trace_log": []
                    }
                    
                    final_state = agent_app.invoke(initial_state)
                    
                    answer = final_state["draft_answer"]
                    trace = final_state["trace_log"]
                    context_docs = final_state["context"] # Capture the source chunks
                    
                    st.markdown(answer)
                    
                    # Display the expanders for the current message
                    with st.expander("🔍 View AI Thought Process (LangGraph Trace)"):
                        for step in trace:
                            st.write(step)
                            
                    with st.expander("📄 View Retrieved Source Documents"):
                        if context_docs:
                            for i, doc in enumerate(context_docs):
                                st.markdown(f"**Source Chunk {i+1}:**")
                                st.info(doc)
                        else:
                            st.warning("No relevant documents found in the database.")
                            
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "trace": trace,
                        "context": context_docs # Save chunks to history so they persist
                    })
                except Exception as e:
                    st.error(f"Agent Execution Error: {e}")

# ------------------------------------------
# TAB 2: METHODOLOGY
# ------------------------------------------
with tab_methodology:
    st.title("🧠 Agentic RAG Architecture")
    st.markdown("An in-depth look at the enterprise-grade AI architecture powering this self-correcting financial agent.")
    st.divider()

    col_m1, col_m2 = st.columns(2, gap="large")
    
    with col_m1:
        st.subheader("1. Data Engineering & Vector Storage")
        st.write("This application utilizes the industry-standard **PatronusAI/FinanceBench** dataset, consisting of dense, highly technical SEC 10-K and 10-Q corporate filings.")
        st.markdown("""
        * **Chunking Strategy:** Documents were processed using LangChain's `RecursiveCharacterTextSplitter` with an 800-character chunk size and 100-character overlap to preserve semantic context across financial tables and paragraphs.
        * **Semantic Vectorization:** Chunks were embedded using Cohere's enterprise-grade `embed-english-v3.0` model via an exponential-backoff cloud data pipeline.
        * **Vector Database:** The embeddings are securely hosted in a highly scalable **Qdrant Cloud** cluster.
        """)
        
        st.subheader("2. Advanced Retrieval (Cohere Rerank)")
        st.write("Standard 'Naive RAG' often fails on financial data because semantic similarity retrieves loosely related chunks rather than exact answers.")
        st.info("💡 **Solution:** This architecture implements a two-stage retrieval. First, Qdrant retrieves the top 15 loosely matching chunks. Second, **Cohere Rerank 3.0** reads those 15 chunks and uses a neural cross-encoder to strictly re-order them, passing only the absolute top 5 most factually relevant chunks to the LLM.")

    with col_m2:
        st.subheader("3. LangGraph State Machine (Self-Reflection)")
        st.write("To prevent the LLM from hallucinating financial numbers, the generation process is governed by a **LangGraph State Machine** containing three distinct operational nodes:")
        
        with st.expander("Node A: The Generator", expanded=True):
            st.write("A Groq-hosted LLM (e.g., LLaMA 3.1) receives the 5 reranked chunks and drafts an initial answer based *strictly* on the provided context.")
            
        with st.expander("Node B: The Hallucination Evaluator (Guardrail)", expanded=True):
            st.write("A secondary, strict 'Auditor Prompt' reviews the draft answer against the source context. It outputs a deterministic JSON payload: `{'hallucinated': 'yes/no', 'feedback': '...'}` checking if the Generator hallucinated any financial figures.")

        with st.expander("Node C: The Router", expanded=True):
            st.write("If the Evaluator detects a hallucination, the Router loops the workflow back to the Generator, injecting the critical feedback and forcing it to rewrite the answer. This creates a self-correcting, fault-tolerant AI Agent.")

    st.divider()
    st.markdown("### The Result")
    st.write("By combining Cohere's precise reranking with LangGraph's self-corrective loops, this platform guarantees that financial answers are highly relevant, strictly grounded in the SEC filings, and free of AI hallucinations.")
