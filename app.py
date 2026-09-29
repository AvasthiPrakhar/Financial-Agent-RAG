"""
Agentic RAG Financial Assistant using LangGraph, Qdrant, Cohere, and Groq.
"""
import os
import json
import streamlit as st
from typing import TypedDict, List
from qdrant_client import QdrantClient
import cohere
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. UI & SECRETS SETUP
# ==========================================
st.set_page_config(page_title="Financial AI Agent", page_icon="📈", layout="wide")

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

# Initialize the primary LLM (Groq LLaMA 3.3 for blazing fast generation)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.1)

# ==========================================
# 2. LANGGRAPH STATE DEFINITION
# ==========================================
# This defines the "memory" of our Agent as it moves through the steps.
class AgentState(TypedDict):
    question: str
    context: List[str]
    draft_answer: str
    feedback: str
    hallucination_found: str  # "yes" or "no"
    iterations: int
    trace_log: List[str] # For the UI to show the thought process

# ==========================================
# 3. AGENT NODES (The Brains of the Operation)
# ==========================================

def retrieve_and_rerank(state: AgentState):
    """Fetches documents from Qdrant and uses Cohere to Rerank the best ones."""
    query = state["question"]
    state["trace_log"].append("🔍 **Step 1: Retrieval & Reranking started...**")
    
    # 1. Semantic Search (Dense Embeddings)
    query_vector = cohere_client.embed(
        texts=[query], model="embed-english-v3.0", input_type="search_query"
    ).embeddings[0]
    
    # THE FIX: Using Qdrant's modern query_points() instead of the deprecated search()
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, 
        query=query_vector, 
        limit=15
    )
    
    # Extract text from the new modern response structure
    docs = [point.payload["text"] for point in search_results.points]
    
    # 2. Cohere Rerank (Boosts accuracy dramatically)
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
    
    # If the evaluator caught a hallucination previously, we force it to fix it!
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
    
    # Use Llama to output JSON
    response = llm.invoke([SystemMessage(content="Output ONLY valid JSON. No markdown tags."), HumanMessage(content=prompt)])
    
    try:
        # Clean the response just in case the LLM wrapped it in markdown
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        eval_result = json.loads(clean_json)
        state["hallucination_found"] = eval_result.get("hallucinated", "yes").lower()
        state["feedback"] = eval_result.get("feedback", "Error parsing feedback.")
    except Exception as e:
        # If parsing fails, we assume it hallucinated to be safe
        state["hallucination_found"] = "yes"
        state["feedback"] = "System failed to parse evaluation. Retrying."

    if state["hallucination_found"] == "yes":
        state["trace_log"].append(f"❌ **Hallucination Detected!** Reason: {state['feedback']}")
    else:
        state["trace_log"].append("✅ **Evaluation Passed!** Answer is factually grounded.")
        
    return state

# ==========================================
# 4. ROUTING LOGIC & GRAPH COMPILATION
# ==========================================
def should_regenerate(state: AgentState):
    """Decides whether to loop back to generation or end the process."""
    if state["hallucination_found"] == "yes" and state["iterations"] < 3:
        return "generate_answer" # Loop back!
    return END # Finish

# Build the LangGraph State Machine
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_and_rerank", retrieve_and_rerank)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("evaluate_hallucination", evaluate_hallucination)

# Define the flow
workflow.add_edge(START, "retrieve_and_rerank")
workflow.add_edge("retrieve_and_rerank", "generate_answer")
workflow.add_edge("generate_answer", "evaluate_hallucination")
workflow.add_conditional_edges("evaluate_hallucination", should_regenerate)

agent_app = workflow.compile()

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.title("📈 Enterprise Financial AI Agent")
st.markdown("*A self-correcting RAG pipeline using LangGraph, Qdrant, and Cohere Rerank. Queries IBM FinQA SEC 10-K Data.*")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace" in msg:
            with st.expander("🔍 View AI Thought Process (LangGraph Trace)"):
                for step in msg["trace"]:
                    st.write(step)

# Handle New User Input
if prompt := st.chat_input("Ask a question about the financial reports (e.g., 'What was the revenue in 2013?'):"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is retrieving, analyzing, and auditing..."):
            initial_state = {
                "question": prompt,
                "context": [],
                "draft_answer": "",
                "feedback": "",
                "hallucination_found": "no",
                "iterations": 0,
                "trace_log": []
            }
            
            # Execute the LangGraph State Machine
            final_state = agent_app.invoke(initial_state)
            
            answer = final_state["draft_answer"]
            trace = final_state["trace_log"]
            
            st.markdown(answer)
            with st.expander("🔍 View AI Thought Process (LangGraph Trace)"):
                for step in trace:
                    st.write(step)
                    
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
