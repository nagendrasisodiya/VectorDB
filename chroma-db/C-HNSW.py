import chromadb
from chromadb.utils import embedding_functions


embedding_fun=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-miniLM-l6-v2"
)
client=chromadb.Client()


# The key configuration parameters are:
#
# space: selects the distance metric. Possible options include:
# l2: squared L2 (Euclidean) distance (default)
# ip: inner (dot) product distance
# cosine: cosine distance
# ef_search: the size of the candidate list used to search for nearest neighbors when a nearest neighbor search is performed.
# The default value is 100.
# Higher values improve both accuracy and recall, but at the cost of slower performance and increased computatuonal cost.
# ef_construction: the size of the candidate list used to select neighbors when a node is inserted during index construction.
# The default value is 100.
# Higher values improve the quality of the index and accuracy, but at the cost of slower performance and increased memory usage.
# max_neighbors: the maximum number of connections each node can have during construction.
# The default value is 16.
# Higher values lead to denser graphs that perform better during searches at the cost of higher memory usage and construction time.
# We can categorize the performance-based parameters into two types:
# ef_search directly controls the breadth of the search at query time, making it the most direct lever for search quality (recall) vs. query speed.
# ef_construction and max_neighbors affect the quality of the built index.
# A higher-quality, denser index (achieved with higher ef_construction and max_neighbors) provides a better foundation for searches, potentially leading to better accuracy.
# However, this quality comes at the cost of significantly longer index build times and higher memory consumption during construction and for storing the index.
collection=client.create_collection(
    name="test-collection",
    metadata={"topic": "hnsw learn"},
    configuration={
        "hnsw":{
            "space":"cosine",
            "ef_search":100,
            "ef_construction":100,
            "max_neighbors":16
        },
        "embedding_function":embedding_fun
    }
)

collection.add(
    documents=[
        "Giant pandas are a bear species that lives in mountainous areas.",
        "A pandas DataFrame stores two-dimensional, tabular data",
        "I think everyone agrees that pandas are some of the cutest animals on the planet",
        "A direct comparison between pandas and polars indicates that polars is a more efficient library than pandas."
    ],
    metadatas=[
        {"topic": "animals"},
        {"topic": "data analysis"},
        {"topic": "animals"},
        {"topic": "data analysis"}
    ],
    ids=["id1", "id2", "id3", "id4"]
)
# Note that querying the database involves passing the query, in a list, to the query_texts parameter in the .query() method.
# The optional parameter n_results controls the number of results to retrieve.
op=collection.query(
    query_texts=['cats'],
    n_results=10
)
print(op)
op=collection.query(
    query_texts=['polar bear'],
    n_results=1
)
print(op) #give unexpected output as it confuse with polars
# to more efficient processing or searching, we can give it more context or use different embedding
op=collection.query(
    query_texts=['polar bear'],
    n_results=1,
    where={"topic":'animals'}
)
print(op)
# we could perform a full - text search to include or exclude documents based on specific words or phrases
op=collection.query(
    query_texts=['bear'],
    n_results=1,
    where_document={'$not_contains':'library'}
)
print(op)

op=collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'},
    where_document={'$not_contains': 'library'}
)
print(op)