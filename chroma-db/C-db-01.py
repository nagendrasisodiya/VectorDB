import chromadb
from chromadb.types import Collection
from chromadb.utils import embedding_functions

embedding_fun=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-l6-v2'
)

client=chromadb.Client()

# creating collection to store data
collection=client.create_collection(
    name='filter_demo',
    metadata={"description":"used for demo filtering"},
    embedding_function= embedding_fun
)
print(f"Collection created: {collection.name}")

# adding documnets in collection

collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

op=collection.get(
    where={"source":{"$eq":"langchain.com"}}
)
print(op)

# finds only documents where the source is "langchain.com" with versions less than 0.3
op=collection.get(
    where={
        "$and":[
            {"source":{"$eq":"langchain.com"}},
            {"version":{"$lt":0.3}}
        ]
    }
)
print(op)

# Filter using Document Content
op=collection.get(
    where_document={"$contains":"pandas"}
)
print(op)


# Combine Metadata and Document Content Filters
op=collection.get(
    where={
        "version":{"$gt":0.1}
    },
    where_document={
        "$or":[
            {"$contains":"LangChain"},
            {"$contains":"Python"}
        ]
    }
)

print(op)