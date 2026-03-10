"""
Multi-Agent LangGraph Pipeline:
  - Retrieval Agent: fetches relevant docs via hybrid search
  - Summarization Agent: synthesizes retrieved docs into an answer
  - Fact-Checking Agent: verifies answer claims against source docs
  - Router: decides which agent to call next
  
LLM: Groq (free, fast) — llama-3.3-70b-versatile
Embeddings: sentence-transformers (free, local)
"""

import os
import json
from typing import TypedDict, List, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from retrieval.hybrid_retriever import HybridRetriever, Document

load_dotenv()


# ──────────────────────────── State ────────────────────────────

class AgentState(TypedDict):
    query: str
    retrieved_docs: List[dict]
    draft_answer: str
    fact_check_result: str
    final_answer: str
    confidence_score: float
    citations: List[dict]
    reasoning_trace: Annotated[List[str], operator.add]
    iteration: int
    error: Optional[str]


# ──────────────────────────── LLM ──────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ──────────────────────────── Agents ───────────────────────────

class RetrievalAgent:
    """Fetches relevant documents using hybrid retrieval."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        trace = [f"[RetrievalAgent] Searching for: '{query}'"]

        try:
            docs = await self.retriever.search(query, k=int(os.getenv("MAX_RETRIEVAL_DOCS", 10)))
            retrieved = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": round(doc.score, 4),
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            trace.append(f"[RetrievalAgent] Retrieved {len(retrieved)} documents.")
        except Exception as e:
            retrieved = []
            trace.append(f"[RetrievalAgent] Error: {e}")

        return {
            **state,
            "retrieved_docs": retrieved,
            "reasoning_trace": trace
        }


class SummarizationAgent:
    """Synthesizes retrieved documents into a coherent answer."""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    async def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = state["retrieved_docs"]
        trace = [f"[SummarizationAgent] Synthesizing answer from {len(docs)} documents."]

        if not docs:
            return {
                **state,
                "draft_answer": "No relevant documents found for this query.",
                "reasoning_trace": trace
            }

        context_parts = []
        for i, doc in enumerate(docs[:8], 1):
            context_parts.append(
                f"[Source {i}: {doc['source']}]\n{doc['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a research assistant synthesizing information from academic papers.
        
Your task:
1. Answer the query comprehensively using ONLY the provided sources
2. Cite sources inline as [Source N]
3. Be precise and factual
4. If sources conflict, acknowledge it
5. If sources don't fully answer the query, say so

Format your answer clearly with:
- A direct answer to the query
- Supporting evidence from sources with inline citations
- A confidence assessment (High/Medium/Low) at the end"""

        user_prompt = f"""Query: {query}

Sources:
{context}

Please synthesize a comprehensive answer."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            draft_answer = response.content
            trace.append("[SummarizationAgent] Draft answer generated.")
        except Exception as e:
            draft_answer = f"Error generating answer: {e}"
            trace.append(f"[SummarizationAgent] Error: {e}")

        return {
            **state,
            "draft_answer": draft_answer,
            "reasoning_trace": trace
        }


class FactCheckingAgent:
    """Verifies claims in the draft answer against source documents."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    async def run(self, state: AgentState) -> AgentState:
        draft = state["draft_answer"]
        docs = state["retrieved_docs"]
        query = state["query"]
        trace = ["[FactCheckingAgent] Verifying claims in draft answer."]

        if not docs or not draft:
            return {
                **state,
                "fact_check_result": "No content to verify.",
                "final_answer": draft,
                "confidence_score": 0.0,
                "citations": [],
                "reasoning_trace": trace
            }

        context = "\n\n".join([
            f"[Source {i+1}: {d['source']}]\n{d['content']}"
            for i, d in enumerate(docs[:6])
        ])

        system_prompt = """You are a fact-checking agent. Your job is to:
1. Verify each claim in the draft answer against the source documents
2. Identify any unsupported or contradicted claims
3. Assign a confidence score (0.0-1.0) based on source support
4. Produce a final, corrected answer

Respond in this EXACT JSON format:
{
  "verified_claims": ["claim 1 is supported by...", ...],
  "unsupported_claims": ["claim X is not found in sources", ...],
  "corrections": ["correction if any", ...],
  "confidence_score": 0.85,
  "final_answer": "The corrected/verified final answer here...",
  "citations": [
    {"source": "paper title", "relevant_quote": "brief quote or description"}
  ]
}"""

        user_prompt = f"""Query: {query}

Draft Answer:
{draft}

Source Documents:
{context}

Verify the draft answer and return JSON."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            raw = response.content.strip()
            # Strip markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)

            fact_check_result = (
                f"Verified: {len(result.get('verified_claims', []))} claims. "
                f"Unsupported: {len(result.get('unsupported_claims', []))} claims."
            )
            final_answer = result.get("final_answer", draft)
            confidence_score = float(result.get("confidence_score", 0.7))
            citations = result.get("citations", [])
            trace.append(f"[FactCheckingAgent] {fact_check_result}")

        except (json.JSONDecodeError, Exception) as e:
            trace.append(f"[FactCheckingAgent] Parse error: {e}. Using draft as final.")
            fact_check_result = "Fact-check parsing failed; using draft answer."
            final_answer = draft
            confidence_score = 0.6
            citations = [{"source": d["source"], "relevant_quote": ""} for d in docs[:3]]

        return {
            **state,
            "fact_check_result": fact_check_result,
            "final_answer": final_answer,
            "confidence_score": confidence_score,
            "citations": citations,
            "reasoning_trace": trace
        }


# ──────────────────────────── Graph ────────────────────────────

def build_rag_graph(retriever: HybridRetriever) -> StateGraph:
    """Assemble the multi-agent LangGraph pipeline."""

    retrieval_agent = RetrievalAgent(retriever)
    summarization_agent = SummarizationAgent()
    fact_checking_agent = FactCheckingAgent()

    async def run_retrieval(state: AgentState) -> AgentState:
        return await retrieval_agent.run(state)

    async def run_summarization(state: AgentState) -> AgentState:
        return await summarization_agent.run(state)

    async def run_fact_check(state: AgentState) -> AgentState:
        return await fact_checking_agent.run(state)

    def should_continue(state: AgentState) -> str:
        """Route: after retrieval, always summarize."""
        if not state.get("retrieved_docs"):
            return "summarize"
        return "summarize"

    def after_summarize(state: AgentState) -> str:
        """Route: after summarization, always fact-check."""
        return "fact_check"

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", run_retrieval)
    graph.add_node("summarize", run_summarization)
    graph.add_node("fact_check", run_fact_check)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", should_continue, {"summarize": "summarize"})
    graph.add_conditional_edges("after_summarize", after_summarize, {"fact_check": "fact_check"})
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "fact_check")
    graph.add_edge("fact_check", END)

    return graph.compile()


async def run_pipeline(query: str, retriever: HybridRetriever) -> AgentState:
    """Run the full RAG pipeline for a query."""
    graph = build_rag_graph(retriever)
    initial_state: AgentState = {
        "query": query,
        "retrieved_docs": [],
        "draft_answer": "",
        "fact_check_result": "",
        "final_answer": "",
        "confidence_score": 0.0,
        "citations": [],
        "reasoning_trace": [f"[Pipeline] Starting query: '{query}'"],
        "iteration": 0,
        "error": None
    }
    result = await graph.ainvoke(initial_state)
    return result
