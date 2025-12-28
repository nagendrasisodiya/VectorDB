# VectorDB

A comprehensive implementation of vector database concepts featuring ChromaDB, sentence transformers, and IBM watsonx.ai integration for semantic search, similarity matching, and AI-powered recommendation systems.

## Overview

This repository demonstrates practical applications of vector databases, from basic similarity search to production-ready RAG (Retrieval Augmented Generation) chatbots. It includes hands-on examples of document embedding, semantic search, metadata filtering, HNSW index optimization, and LLM-powered intelligent recommendations.

## Features

### Core Vector Database Operations
- **Document Embedding**: Convert text into vector representations using sentence transformers
- **Semantic Search**: Query documents based on meaning rather than exact keyword matches
- **Advanced Filtering**: Combine metadata and content-based filters with logical operators
- **HNSW Configuration**: Fine-tune index parameters for optimal search performance
- **Distance Metrics**: Compare L2, cosine, and inner product similarity measures

### Advanced Applications
- **RAG-Powered Chatbot**: Food recommendation system with IBM watsonx.ai Granite model integration
- **Employee Search System**: Intelligent employee record retrieval with multi-criteria filtering
- **Similarity Search Engine**: Grocery item search with semantic understanding

## Installation

### Prerequisites
- Python 3.12 or 3.13
- Poetry (recommended) or pip

### Using Poetry
```bash
# Install all dependencies
poetry install

# Or add dependencies manually
poetry add sentence-transformers==4.1.0
poetry add chromadb>=1.4.0
poetry add ibm-watsonx-ai>=1.4.11
poetry add load-dotenv==0.1.0
```

### Using pip
```bash
pip install sentence-transformers==4.1.0
pip install chromadb>=1.4.0
pip install numpy>=2.4.0
pip install scipy>=1.16.3
pip install ibm-watsonx-ai>=1.4.11
pip install load-dotenv==0.1.0
```

## Project Structure

```
VectorDB/
├── chroma-db/
│   ├── C-db-01.py              # Metadata and content filtering examples
│   └── C-HNSW.py               # HNSW index configuration and querying
│
├── food search boot/
│   ├── RAG_chat_bot.py         # RAG-powered food recommendation chatbot
│   ├── search.py               # Basic food search functionality
│   ├── shared_functions.py    # Shared utility functions
│   ├── FoodDataSet.json        # Food database
│   └── .env                    # API keys and credentials
│
├── Similarity Search/
│   ├── Employee Records search.py  # Employee search system
│   └── s-01.py                     # Basic similarity search demo
│
├── V-DB-01.py                  # Manual distance calculation and embeddings
├── pyproject.toml              # Project dependencies
└── README.md
```

## Usage Examples

### 1. Basic Similarity Search
```python
# Run the grocery similarity search demo
python s-01.py
```

### 2. Advanced Filtering with ChromaDB
```python
# Demonstrates metadata and document content filtering
python C-db-01.py
```

### 3. HNSW Index Configuration
```python
# Learn about HNSW parameters and their impact
python C-HNSW.py
```

### 4. Employee Search System
```python
# Search employees by skills, experience, location
python "Employee Records search.py"
```

### 5. RAG Food Recommendation Chatbot
```bash
# Set up environment variables first
# Create a .env file with:
# WATSONX_API_KEY=your_api_key
# WATSONX_PROJECT_ID=your_project_id

# Run the chatbot
cd "food search boot"
python RAG_chat_bot.py
```

Example queries for the chatbot:
- "I want something spicy and healthy for dinner"
- "What Italian dishes do you recommend under 400 calories?"
- "Suggest some protein-rich breakfast options"

## Key Concepts

### Embeddings
Documents are converted into dense vector representations using the `all-MiniLM-L6-v2` model, enabling semantic similarity comparisons beyond simple keyword matching.

### Distance Metrics
- **L2 (Euclidean)**: Measures straight-line distance between vectors
- **Cosine**: Measures angle between vectors, ideal for text similarity
- **Inner Product**: Dot product similarity, useful for dense representations

### HNSW Parameters
- **space**: Distance metric (l2, cosine, ip)
- **ef_search**: Search breadth (default: 100) - higher = better accuracy, slower queries
- **ef_construction**: Index build quality (default: 100) - higher = better index, longer build time
- **max_neighbors**: Maximum connections per node (default: 16) - higher = denser graph, more memory

### Filtering Operations
ChromaDB supports complex queries with:
- Metadata filters: `where` with operators like `$eq`, `$lt`, `$gt`, `$gte`, `$in`
- Document filters: `where_document` with `$contains`, `$not_contains`
- Logical operators: `$and`, `$or` for combining conditions

### RAG Architecture
The food recommendation chatbot demonstrates:
1. **Vector Search**: Finds relevant food items from the database
2. **Context Preparation**: Formats search results for LLM consumption
3. **LLM Generation**: Uses IBM Granite model to generate personalized recommendations
4. **Fallback Strategy**: Provides basic recommendations if LLM fails

## Configuration

### Environment Variables
Create a `.env` file in the `food search boot/` directory:
```env
WATSONX_API_KEY=your_ibm_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id
GOOGLE_API_KEY=your_google_api_key  # Optional, for future features
```

### IBM watsonx.ai Setup
1. Create an IBM Cloud account
2. Provision a Watson Machine Learning service
3. Create a watsonx.ai project
4. Associate the WML service with your project
5. Copy your API key and project ID to `.env`

## Features by File

### C-db-01.py
- Metadata filtering with comparison operators
- Document content filtering
- Combined metadata + content filters
- Logical operators (`$and`, `$or`)

### C-HNSW.py
- HNSW index configuration
- Performance tuning parameters
- Query optimization techniques
- Full-text search with filters

### RAG_chat_bot.py
- IBM watsonx.ai Granite model integration
- Vector similarity search
- Context-aware recommendations
- Conversation history management
- Fallback response generation
- Interactive chatbot interface

### Employee Records search.py
- Multi-criteria employee search
- Skills-based matching
- Experience and location filtering
- Combined similarity + metadata queries

### s-01.py
- Basic ChromaDB setup
- Simple similarity search
- Collection management

## Dependencies

```toml
sentence-transformers = "4.1.0"
chromadb = ">=1.4.0,<2.0.0"
numpy = ">=2.4.0,<3.0.0"
scipy = ">=1.16.3,<2.0.0"
ibm-watsonx-ai = ">=1.4.11,<2.0.0"
load-dotenv = "0.1.0"
```

## Learning Path

1. **Start with basics**: Run `s-01.py` to understand similarity search
2. **Explore filtering**: Try `C-db-01.py` for metadata and content filters
3. **Optimize performance**: Experiment with `C-HNSW.py` parameters
4. **Build applications**: Study `Employee Records search.py` for real-world use
5. **Advanced RAG**: Deploy the food chatbot to see LLM integration

## Future Enhancements

- [ ] Multi-modal embedding support (text + images)
- [ ] Advanced RAG with re-ranking
- [ ] Vector database persistence
- [ ] REST API for chatbot deployment
- [ ] Evaluation metrics dashboard
- [ ] Query caching for improved performance

## Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Share optimization techniques
- Contribute example use cases

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [IBM watsonx.ai Docs](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-overview.html)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)

