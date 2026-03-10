"""
Document ingestion: load PDFs/text files and chunk them into Documents.
"""

import os
import uuid
from typing import List
from pathlib import Path

from retrieval.hybrid_retriever import Document


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_txt_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return ""


def ingest_documents(papers_dir: str, chunk_size: int = 512, overlap: int = 64) -> List[Document]:
    """
    Load all PDF and TXT files from papers_dir, chunk them,
    and return a list of Document objects.
    """
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        os.makedirs(papers_path, exist_ok=True)
        print(f"Created papers directory: {papers_path}")
        # Create sample documents for demo
        return _create_sample_documents()

    documents = []
    files = list(papers_path.glob("*.pdf")) + list(papers_path.glob("*.txt"))

    if not files:
        print("No files found in papers directory. Using sample documents.")
        return _create_sample_documents()

    for filepath in files:
        print(f"Loading: {filepath.name}")
        if filepath.suffix.lower() == ".pdf":
            text = load_pdf_file(str(filepath))
        else:
            text = load_txt_file(str(filepath))

        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=str(uuid.uuid4()),
                content=chunk,
                metadata={
                    "source": filepath.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "filepath": str(filepath)
                }
            )
            documents.append(doc)

    print(f"Ingested {len(documents)} chunks from {len(files)} files.")
    return documents


def _create_sample_documents() -> List[Document]:
    """Create sample research documents for demonstration."""
    samples = [
        {
            "title": "Attention Is All You Need",
            "content": """The Transformer architecture introduced in 'Attention Is All You Need' by Vaswani et al. 
            relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. 
            The model uses multi-head self-attention to process sequences in parallel, achieving state-of-the-art 
            results on machine translation tasks. The architecture consists of an encoder-decoder structure where 
            the encoder maps input sequences to continuous representations, and the decoder generates output sequences 
            auto-regressively. Positional encodings are added to inject sequence order information since the model 
            contains no recurrence or convolution."""
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "content": """BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train 
            deep bidirectional representations from unlabeled text by jointly conditioning on both left and right 
            context in all layers. BERT uses two pre-training tasks: Masked Language Model (MLM) and Next Sentence 
            Prediction (NSP). In MLM, some percentage of the input tokens are masked at random, and the model 
            predicts those masked tokens. NSP helps the model understand sentence relationships. Fine-tuned BERT 
            models achieved state-of-the-art results on eleven NLP tasks including GLUE, MultiNLI, and SQuAD."""
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "content": """GPT-3 is an autoregressive language model with 175 billion parameters, trained on 
            a dataset of hundreds of billions of tokens. The model demonstrates strong few-shot learning 
            capabilities, performing tasks by conditioning on a few examples in the prompt without any gradient 
            updates. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, 
            and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, 
            such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic."""
        },
        {
            "title": "RAG: Retrieval-Augmented Generation",
            "content": """Retrieval-Augmented Generation (RAG) combines parametric and non-parametric memory for 
            language generation. A pre-trained seq2seq model acts as the parametric memory and a dense vector 
            index of Wikipedia acts as non-parametric memory. These are accessed with a pre-trained neural 
            retriever. RAG models retrieve relevant documents using Maximum Inner Product Search (MIPS) and 
            use them as additional context when generating outputs. This approach allows models to access and 
            use information that was not in their training data, making them more factually accurate and 
            enabling knowledge updates without full retraining."""
        },
        {
            "title": "Chain-of-Thought Prompting",
            "content": """Chain-of-thought prompting enables large language models to perform complex multi-step 
            reasoning. By providing a few examples that include intermediate reasoning steps, models can 
            decompose complex problems into smaller steps and solve them sequentially. This technique 
            significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks. 
            Zero-shot chain-of-thought prompting, using phrases like 'Let's think step by step', also 
            elicits reasoning without task-specific examples. The emergent ability appears primarily in 
            models with more than 100 billion parameters."""
        },
        {
            "title": "LangGraph: Building Stateful Multi-Actor Applications",
            "content": """LangGraph is a library for building stateful, multi-actor applications with LLMs, 
            used to create agent and multi-agent workflows. Unlike simple chains, LangGraph supports cycles, 
            which are crucial for agentic behaviors where the model must decide the next action repeatedly. 
            It extends LangChain Expression Language with coordination of multiple chains or actors across 
            multiple compute steps using cycles. LangGraph uses a graph-based approach where nodes represent 
            computation steps and edges define the flow between them, with support for conditional routing 
            based on state."""
        },
    ]

    documents = []
    for sample in samples:
        doc = Document(
            id=str(uuid.uuid4()),
            content=sample["content"],
            metadata={"source": sample["title"], "chunk_index": 0, "total_chunks": 1}
        )
        documents.append(doc)

    print(f"Created {len(documents)} sample documents.")
    return documents
