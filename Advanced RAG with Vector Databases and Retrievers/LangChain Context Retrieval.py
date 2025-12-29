import logging
import os

from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams, EmbedTextParamsMetaNames
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import MultiQueryRetriever, SelfQueryRetriever, ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_ibm import WatsonxEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

load_dotenv(dotenv_path=r"C:\Users\ASUS\OneDrive\Desktop\GEN-AI\VectorDB\food search boot\.env")
def llm():
    model_id='meta-llama/llama-3-3-70b-instruct'
    parameters={
        GenParams.MAX_NEW_TOKENS:250,
        GenParams.TEMPERATURE:0.5
    }
    project_id = "skills-network"
    credentials = {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": os.getenv("WATSONX_API_KEY")
    }

    model=ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )

    mixtral_llm = WatsonxLLM(model=model)
    return mixtral_llm

def text_splitter(data, chunk_size, chunk_overlap):
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks=textsplitter.split_documents(data)
    return chunks

def watsonx_embedding():
    embed_params={
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
    }
    watsonx_embedding=WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://eu-de.ml.cloud.ibm.com",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=embed_params,
    )
    return  watsonx_embedding

loader=TextLoader(r"C:\Users\ASUS\OneDrive\Desktop\GEN-AI\VectorDB\Advanced RAG with Vector Databases and Retrievers\companypolicies.txt")
text_data=loader.load()

chunk_text=text_splitter(text_data, 200, 20)

vectordb=Chroma.from_documents(chunk_text, watsonx_embedding())

query="Smoking Policy"
retriever=vectordb.as_retriever()

docs=retriever.invoke(query)
print(docs)

# MMR Search
#
# MMR in vector stores is a technique used to balance the relevance and diversity of retrieved results.
# It selects documents that are both highly relevant to the query and minimally similar to previously selected documents.
# his approach helps to avoid redundancy and ensures a more comprehensive coverage of different aspects of the query.

retriever=vectordb.as_retriever( search_type='mmr')
docs=retriever.invoke(query)
print("mmr query result: ")
print(docs)


# Multi-Query Retrieve
# Distance-based vector database retrieval represents queries in high-dimensional space and finds similar embedded documents based on "distance".
# However, retrieval results may vary with subtle changes in query wording or if the embeddings do not accurately capture the data's semantics.
#
# The MultiQueryRetriever addresses this by using an LLM to generate multiple queries from different perspectives for a given user input query.
# For each query, it retrieves a set of relevant documents and then takes the unique union of these results to form a larger set of potentially relevant documents.

loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
pdf_data = loader.load()

# print(pdf_data[1])
chunks_pdf=text_splitter(pdf_data, 500, 200)

ids = vectordb.get()["ids"]
vectordb.delete(ids) #  delete existing embeddings from previous documents and then store current document embeddings in.
vectordb=Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())

query= "What does the paper say about langchain?"

retriever=MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm()
)

docs=retriever.invoke(query)
print("Multi Query Retriever results:")
print(docs)


# Self-Querying Retriever
#
# A Self-Querying Retriever, as the name suggests, has the ability to query itself.
# Specifically, given a natural language query, the retriever uses a query-constructing LLM chain to generate a structured query.
# It then applies this structured query to its underlying vector store.
# This enables the retriever to not only use the user-input query for semantic similarity comparison with the contents of stored documents but also to extract and apply filters based on the metadata of those documents.

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
vectordb=Chroma.from_documents(docs, watsonx_embedding())
document_content_description = "Brief summary of a movie."

retriever=SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info
)

op=retriever.invoke("I want to watch a movie rated higher than 8.5")
print(" Self query retriever op-01: ")
print(op)
retriever.invoke("Has Greta Gerwig directed any movies about women")
print(" Self query retriever op-02: ")
print(op)

# Parent Document Retrieve
#
# When splitting documents for retrieval, there are often conflicting desires:
#
# You may want to have small documents so that their embeddings can most accurately reflect their meaning.
# If the documents are too long, the embeddings can lose meaning.
#
# You want to have long enough documents so that the context of each chunk is retained.
#
# The ParentDocumentRetriever strikes that balance by splitting and storing small chunks of data.
# During retrieval, it first fetches the small chunks but then looks up the parent IDs for those chunks and returns those larger documents.

parent_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
child_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=20, separator='\n')

vectordb=Chroma(
    collection_name="split_parents",
    embedding_function=watsonx_embedding()
)
store=InMemoryStore()

retriever=ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)
retriever.add_documents(text_data)
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)