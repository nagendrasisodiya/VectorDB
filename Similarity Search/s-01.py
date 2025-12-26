import chromadb
from chromadb.utils import embedding_functions
from numpy.ma.core import ids

embedding_fun=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client=chromadb.Client()

collection_name="my_grocery_collection"


# function for similarity search
def  perform_similarity_search(collection, all_items):
    try:
        query_term='apple'
        results=collection.query(
            query_texts=[query_term],
            n_results=3
        )
        print(f"Query results for '{query_term}':")
        print(results)
        pass
    except Exception as error:
        print(f" Exception in similarity search:{error}")


# main function for interacting with chromadb
def main():
    global collection
    try:
        collection=client.create_collection(
            name=collection_name,
            metadata={'description':'collection for storing grocery data'},
            configuration={
                "hnsw":{"space": "cosine"},
                "embedding_function":embedding_fun
            }
        )
        pass
    except Exception as error:
        print(f"Error :{error}")
    print(f"Collection created :{collection.name}")
    texts = [
        'fresh red apples',
        'organic bananas',
        'ripe mangoes',
        'whole wheat bread',
        'farm-fresh eggs',
        'natural yogurt',
        'frozen vegetables',
        'grass-fed beef',
        'free-range chicken',
        'fresh salmon fillet',
        'aromatic coffee beans',
        'pure honey',
        'golden apple',
        'red fruit'
    ]
    ids=[f"food_{index+1} " for index,_ in enumerate(texts)]
    collection.add(
        documents=texts,
        metadatas=[{"source": "grocery_store", "category":"food"} for _ in texts],
        ids=ids
    )
    all_items=collection.get()
    print(f"all the items in grocery: {all_items}")

    perform_similarity_search(collection, all_items)

if __name__ == "__main__":
    main()