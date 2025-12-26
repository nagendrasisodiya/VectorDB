# VectorDB

A practical implementation of vector database concepts using ChromaDB and sentence transformers for semantic search and similarity matching.

## Overview

A learning-focused exploration of vector database operations including document embedding, similarity search, metadata filtering, and HNSW (Hierarchical Navigable Small World) index configuration. This repository contains practical examples and experiments with ChromaDB and sentence transformers.

## Features

- **Document Embedding**: Convert text documents into vector representations using sentence transformers
- **Semantic Search**: Query documents based on meaning rather than exact keyword matches
- **Advanced Filtering**: Combine metadata and content-based filters for precise retrieval
- **HNSW Configuration**: Fine-tune index parameters for optimal search performance
- **Distance Metrics**: Compare different similarity measures (L2, cosine, inner product)

## Installation

This project requires Python 3.12 or 3.13. Install dependencies using Poetry:

```bash
# Install from pyproject.toml
poetry install

# Or add dependencies manually
poetry add sentence-transformers==4.1.0
poetry add chromadb>=1.4.0
```

Alternatively, using pip:

```bash
pip install sentence-transformers==4.1.0 chromadb>=1.4.0
```

## Project Structure

```
VectorDB/
├── C-db-01.py          # Metadata and content filtering examples
├── C-HNSW.py           # HNSW index configuration and querying
├── V-DB-01.py          # Manual distance calculation and embeddings
├── pyproject.toml      # Project dependencies
└── README.md
```

## Key Concepts

### Embeddings

Text documents are converted into dense vector representations using the `all-MiniLM-L6-v2` model, enabling semantic similarity comparisons.

### Distance Metrics

- **L2 (Euclidean)**: Measures straight-line distance between vectors
- **Cosine**: Measures angle between vectors, ignoring magnitude
- **Inner Product**: Dot product similarity measure

### Filtering

ChromaDB supports complex queries combining metadata filters (`where`) and document content filters (`where_document`) with logical operators like `$and`, `$or`, `$eq`, `$lt`, `$gt`, and `$contains`.

## Requirements

- Python 3.12 or 3.13
- sentence-transformers 4.1.0
- chromadb >=1.4.0

## Contributing

This is a personal learning project, but suggestions and feedback are welcome.

## License

This project is available for educational purposes.