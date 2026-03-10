# 🔬 Multi-Agent RAG Research Assistant

This project leverages a multi-agent LLM pipeline to analyze and query across thousands of research papers. It enables researchers, students, and analysts to retrieve relevant documents, synthesize answers with citations, and verify facts automatically using a 3-agent LangGraph architecture — including a Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion (RRF) for state-of-the-art retrieval accuracy.

---

## 🚀 Demo

> 📹 [Click here to watch the demo](#) (https://drive.google.com/file/d/1aewtYXCiDkR9mcGuVKAMaofZ7bzDhw9V/view?usp=drive_link)

---

## 📸 Project Screenshots

> https://drive.google.com/file/d/1iL05Etoyw5u8cjCbQ7tBb47J0acSRLpm/view?usp=drive_link
|

---

## 🏗️ Architecture

```
User Query
    │
    ▼
RetrievalAgent
├── FAISS dense search  (sentence-transformers, local)
├── BM25 sparse search  (keyword matching)
└── RRF fusion          (Reciprocal Rank Fusion)
    │
    ▼
SummarizationAgent  →  ChatGroq (llama-3.3-70b-versatile)
    │
    ▼
FactCheckingAgent   →  Confidence Score + Citations
    │
    ▼
FastAPI  →  Streamlit UI
```

---

## 🛠️ Installation Steps

**1. Clone the repository**

```bash
git clone https://github.com/mrinal1404/Multi-Agent-RAG-Research-Assistant.git
cd Multi-Agent-RAG-Research-Assistant
```

**2. Create and activate virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and add your free Groq API key — get one at [console.groq.com](https://console.groq.com):

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> ⚠️ Never commit your `.env` file to GitHub. It is already listed in `.gitignore`.

**5. (Optional) Add your own research papers**

Drop any `.pdf` or `.txt` files into `data/papers/`. If left empty, 6 built-in sample papers are used automatically (Transformer, BERT, GPT-3, RAG, LangGraph, Chain-of-Thought).

**6. Start the backend** *(Terminal 1)*

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Wait for: `Application startup complete.`

**7. Start the UI** *(Terminal 2)*

```bash
streamlit run ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ How to Use

1. Type a research query in the search box
   - *"How does BERT use masked language modeling?"*
   - *"What is Reciprocal Rank Fusion?"*
   - *"Explain the Transformer attention mechanism"*
2. Toggle **⚡ Stream Response** for real-time token streaming
3. Click **🚀 Search**
4. View the answer along with:
   - 📎 **Citations** — exactly which papers each claim came from
   - 🟢 **Confidence Score** — how well the answer is supported by sources
   - 🧠 **Agent Reasoning Trace** — step-by-step agent decisions
5. Upload new papers from the sidebar and click **Rebuild Index**

---

## 💻 Built With

Technologies used in the project:

* LangGraph
* LangChain
* Groq (LLM — free, llama-3.3-70b-versatile)
* sentence-transformers (all-MiniLM-L6-v2)
* FAISS
* BM25 (rank-bm25)
* FastAPI
* Streamlit
* NumPy
* PyPDF

---

## 📁 Project Structure

```
rag_research_assistant/
├── agents/
│   └── rag_pipeline.py       # LangGraph 3-agent pipeline
├── retrieval/
│   ├── hybrid_retriever.py   # FAISS + BM25 + RRF
│   └── ingestion.py          # PDF/TXT loader and chunker
├── api/
│   └── main.py               # FastAPI backend (streaming SSE)
├── ui/
│   └── app.py                # Streamlit frontend
├── data/
│   └── papers/               # Drop your PDFs here
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🐛 Common Issues

| Problem | Fix |
|---------|-----|
| FAISS won't install on Windows | `pip install faiss-cpu==1.7.4` |
| `GROQ_API_KEY` error | Make sure `.env` exists and restart uvicorn after editing it |
| Embedding model slow on first run | Normal — downloads ~90MB once then cached forever |
| Port already in use | Change `--port 8000` to `--port 8002` |
| `ModuleNotFoundError` | Make sure venv is activated and you are in the project root folder |

---

## 📄 License

MIT License — free to use and modify.
