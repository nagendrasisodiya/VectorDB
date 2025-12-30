import os
import json
from typing import List, Optional
import asyncio
import warnings

import Stemmer
import numpy as np
from dotenv import load_dotenv
from huggingface_hub.hf_api import api

warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

# IBM WatsonX LlamaIndex integration
from ibm_watsonx_ai import APIClient
from llama_index.llms.ibm import WatsonxLLM

# Sentence transformers
from sentence_transformers import SentenceTransformer

# Statistical libraries for fusion techniques
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available - some advanced fusion features will be limited")



load_dotenv(dotenv_path=r"C:\Users\ASUS\OneDrive\Desktop\GEN-AI\VectorDB\food search boot\.env")


def create_watsonx_llm():
    try:
        api_client = APIClient({
            'url': "https://eu-de.ml.cloud.ibm.com",
            'apikey':os.getenv("WATSONX_API_KEY")
        })
        # Use llama-index-llms-ibm (official watsonx.ai integration)
        llm = WatsonxLLM(
            model_id="ibm/granite-3-3-8b-instruct",
            url="https://eu-de.ml.cloud.ibm.com",
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            api_client=api_client,
            temperature=0.9
        )
        print("watsonx.ai LLM initialized using official LlamaIndex integration")
        return llm
    except Exception as error:
        print(f"⚠️ watsonx.ai initialization error: {error}")
        print("Falling back to mock LLM for demonstration")

embed_model=HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm=create_watsonx_llm()

# Configure global settings
# configuring global settings means defining the default AI models our system will use everywhere, without needing to re-specify them each time.
Settings.llm = llm
Settings.embed_model = embed_model
print("watsonx.ai LLM and embeddings configured!")


# Sample data
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning",
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}


class AdvancedRetrievers:
    def __init__(self):
        self.documents=[Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes=SentenceSplitter().get_nodes_from_documents(self.documents)

        print("creating indexes")
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)


init_setup=AdvancedRetrievers()

# vector store retriever
print("=" * 60)
print("1. VECTOR INDEX RETRIEVER")
print("=" * 60)

vector_retriever=VectorIndexRetriever(
    index=init_setup.vector_index,
    similarity_top_k=3
)

# Alternative creation method
# alt_retriever = init_setup.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["basic"]  # "What is machine learning?"
nodes=vector_retriever.retrieve(query)

print(f"Query: {query}")
print(f"retrieved {len(nodes)} nodes: ")
for i, node in enumerate(nodes, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:100]}...")
    print()


# BM25 RETRIEVER
print("=" * 60)
print("2. BM25 RETRIEVER")
print("=" * 60)

try:

    bm25_retriever=BM25Retriever.from_defaults(
        nodes=init_setup.nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
    query = DEMO_QUERIES["technical"]  # "neural networks deep learning"
    nodes = bm25_retriever.retrieve(query)

    print(f"Query: {query}")
    print("BM25 analyzes exact keyword matches with sophisticated scoring")
    print(f"Retrieved {len(nodes)} nodes:")

    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0
        print(f"{i}. BM25 Score: {score:.4f}")
        print(f"   Text: {node.text[:100]}...")

        # Highlight which query terms appear in the text
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            print(f"   → Found terms: {found_terms}")
        print()

except ImportError:
    print("⚠️ BM25Retriever requires 'poetry install PyStemmer'")
    print("Demonstrating BM25 concepts with fallback vector search...")

    fallback_retriever = init_setup.vector_index.as_retriever(similarity_top_k=3)
    query = DEMO_QUERIES["technical"]
    nodes = fallback_retriever.retrieve(query)

    print(f"Query: {query}")
    print("(Using vector fallback to demonstrate BM25 concepts)")

    for i, node in enumerate(nodes, 1):
        print(f"{i}. Vector Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")

        # Demonstrate TF-IDF concept manually
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]

        if found_terms:
            print(f"   → BM25 would boost this result for terms: {found_terms}")
        print()

# DOCUMENT SUMMARY INDEX RETRIEVERS
print("=" * 60)
print("3. DOCUMENT SUMMARY INDEX RETRIEVERS")
print("=" * 60)

# LLM-based document summary retriever
doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
    init_setup.document_summary_index,
    choice_top_k=3
)

# Embedding-based document summary retriever
doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
    init_setup.document_summary_index,
    similarity_top_k=3
)

query = DEMO_QUERIES["learning_types"]  # "different types of learning"

print(f"Query: {query}")

print("\nA) LLM-based Document Summary Retriever:")
print("Uses LLM to select relevant documents based on summaries")
try:
    nodes_llm = doc_summary_retriever_llm.retrieve(query)
    print(f"Retrieved {len(nodes_llm)} nodes")
    for i, node in enumerate(nodes_llm[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"LLM-based retrieval demo: {str(e)[:100]}...")

print("B) Embedding-based Document Summary Retriever:")
print("Uses vector similarity between query and document summaries")
try:
    nodes_emb = doc_summary_retriever_embedding.retrieve(query)
    print(f"Retrieved {len(nodes_emb)} nodes")
    for i, node in enumerate(nodes_emb[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"Embedding-based retrieval demo: {str(e)[:100]}...")



# AUTO MERGING RETRIEVER
print("=" * 60)
print("4. AUTO MERGING RETRIEVER")
print("=" * 60)

