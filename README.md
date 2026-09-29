# 📈 Enterprise Financial AI Agent (Agentic RAG)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-005073?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-D3215D?style=for-the-badge&logo=qdrant&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-3959A8?style=for-the-badge&logo=cohere&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

### 🔗 Quick Links
*   **🔴 Live Audit Terminal:** [Click here to view the live Application](https://prakhar-finance-agent.streamlit.app/)
*   **💼 LinkedIn:** [Prakhar Avasthi](http://www.linkedin.com/in/prakhar-avasthi-35067a1bb)
*   **📧 Email:** prakharavasthi1999@gmail.com
*   **🐙 GitHub:** [AvasthiPrakhar](https://github.com/AvasthiPrakhar)

---

## 📌 Project Overview
Standard "Naive RAG" systems are heavily prone to hallucinations, making them dangerous for deployment in the financial sector. 

This project is an **Enterprise-Grade Agentic RAG Platform** built to query dense SEC 10-K and 10-Q corporate filings. It utilizes a **self-correcting LangGraph State Machine** to iteratively audit, catch, and correct its own hallucinations before presenting financial data to the user.

## 🏗️ The Agentic Workflow (LangGraph)
The core of this application is a multi-node AI agent designed for strict factual grounding:

1. **Context Reformulator:** Intercepts conversational follow-up questions and utilizes session history to rewrite vague pronouns into precise, standalone database search queries.
2. **Two-Stage Retrieval:** 
   * **Semantic Search:** Queries a **Qdrant Vector Database** to retrieve the top 15 loosely matching chunks.
   * **Neural Reranking:** Passes results to **Cohere Rerank 3.0**, utilizing a cross-encoder model to strictly filter and re-order the top 5 most factually relevant chunks.
3. **The Generator:** A dynamically fetched LLM (e.g., Groq LLaMA 3.1 / 3.3) drafts an initial answer based *strictly* on the provided context.
4. **The Hallucination Evaluator (Guardrail):** A secondary strict "Auditor Prompt" reviews the draft answer against the source context, outputting a deterministic JSON payload to flag ungrounded financial figures.
5. **The Router:** If a hallucination is detected, the router intercepts the output and loops the workflow back to the Generator, injecting critical feedback and forcing a rewrite until the answer is 100% grounded.

## 🗄️ Data Engineering Pipeline
The backend vector database was populated using a custom exponential-backoff ETL pipeline.
*   **Dataset:** `virattt/financebench` (Industry-standard SEC filing benchmark for RAG).
*   **Processing:** Chunked via LangChain's `RecursiveCharacterTextSplitter` (800-char chunks, 100-char overlap).
*   **Vectorization:** Embeddings generated using Cohere's enterprise `embed-english-v3.0` model.
