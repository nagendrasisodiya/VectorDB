import re

import faiss
import numpy as np
import pandas as pd
import tensorflow_hub as hub
# Direct download URL for BBC News dataset
url = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
print("Downloading BBC News dataset...")

try:
    # Load the dataset
    df = pd.read_csv(url)

    print("✓ Dataset loaded successfully!")
    print(f"Total articles: {len(df)}")
    print(f"Categories: {df['category'].unique().tolist()}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("\n" + "=" * 80)

# Display first 3 articles
for i in range(3):
    print(f"\nSample article {i + 1}:")
    print(f"Category: {df['category'][i]}")
    print(f"Text: {df['text'][i][:200]}...")  # First 200 characters
    print("-" * 80)


# Pre-processing Data
# Steps in Preprocessing:

# Fetching Data:
# We load the complete 20 Newsgroups dataset using fetch_20newsgroups from sklearn.datasets.
# documents = newsgroups.data stores all the newsgroup documents in a list.

# Defining the Preprocessing Function:
# The preprocess_text function is designed to clean each text document. Here's what it does to every piece of text:
# Removes Email Headers: Strips off lines that start with 'From:' as they usually contain metadata like email addresses.
# Eliminates Email Addresses: Finds patterns resembling email addresses and removes them.
# Strips Punctuations and Numbers: Removes all characters except alphabets, aiding in focusing on textual data.
# Converts to Lowercase: Standardizes the text by converting all characters to lowercase, ensuring uniformity.
# Trims Excess Whitespace: Cleans up any extra spaces, tabs, or line breaks.

# Applying Preprocessing:
# We iterate over each document in the documents list and apply our preprocess_text function.
# The cleaned documents are stored in processed_documents, ready for further processing.
# Preprocessing function

def preprocess_text(text):
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Preprocess all documents
print("\nPreprocessing documents...")
documents = df['text'].tolist()
processed_documents = [preprocess_text(doc) for doc in documents]


print(f"✓ Preprocessing complete!")


# Printing the original article
print("Original article:\n")
print(documents[0])
print("\n" + "-"*80 + "\n")

# Printing the preprocessed article
print("Preprocessed article:\n")
print(processed_documents[0])
print("\n" + "-"*80 + "\n")

print(f"✓ Total documents processed: {len(processed_documents)}")


# Universal Sentence Encoder


# After preprocessing the text data, the next step is to transform this cleaned text into numerical vectors using the Universal Sentence Encoder (USE).
# These vectors capture the semantic essence of the text.
#
# Loading the USE Module:
# We use TensorFlow Hub (hub) to load the pre-trained Universal Sentence Encoder.
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") fetches the USE module, making it ready for vectorization.

# Defining the Embedding Function:
# The embed_text function is defined to take a piece of text as input and return its vector representation.
# Inside the function, embed(text) converts the text into a high-dimensional vector, capturing the nuanced semantic meaning.
# .numpy() is used to convert the result from a TensorFlow tensor to a NumPy array, which is a more versatile format for later operations.

# Vectorizing Preprocessed Documents:
# We then apply the embed_text function to each document in our preprocessed dataset, processed_documents.
# np.vstack([...]) stacks the vectors vertically to create a 2D array, where each row represents a document.
# The resulting array X_use holds the vectorized representations of all the preprocessed documents, ready to be used for semantic search indexing and querying.
# By vectorizing the text with USE, we've now converted our textual data into a format that can be efficiently processed by machine learning algorithms, setting the stage for the next step: indexing with FAISS.

# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_text(text):
    return embed(text).numpy()

# Generate embeddings for each preprocessed document
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])
print('done')

# Indexing with FAISS
#
# With our documents now represented as vectors using the Universal Sentence Encoder, the next step is to use FAISS (Facebook AI Similarity Search) for efficient similarity searching.

# Creating a FAISS Index:
# We first determine the dimension of our vectors from X_use using X_use.shape[1].
# A FAISS index (index) is created specifically for L2 distance (Euclidean distance) using faiss.IndexFlatL2(dimension).
# We add our document vectors to this index with index.add(X_use).
# This step effectively creates a searchable space for our document vectors.

# Choosing the Right Index:
# In this project, we use IndexFlatL2 for its simplicity and effectiveness in handling small to medium-sized datasets.
# FAISS offers a variety of indexes tailored for different use cases and dataset sizes.
# Depending on your specific needs and the complexity of your data, you might consider other indexes for more efficient searching.
# For larger datasets or more advanced use cases, indexes like IndexIVFFlat, IndexIVFPQ, and others can provide faster search times and reduced memory usage.

dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)  # Creating a FAISS index
index.add(X_use)  # Adding the document vectors to the index

# Querying with FAISS

# Defining the Search Function:
# The search function is designed to find documents that are semantically similar to a given query.
# It preprocesses the query text using the preprocess_text function to ensure consistency.
# The query text is then converted to a vector using embed_text.
# FAISS performs a search for the nearest neighbors (k) to this query vector in our index.
# It returns the distances and indices of these nearest neighbors.

# Executing a Query and Displaying Results:
# We test our search engine with an example query (e.g., "motorcycle").
# The search function returns the indices of the documents in the index that are most similar to the query.
# For each result, we display:
# The ranking of the result (based on distance).
# The distance value itself, indicating how close the document is to the query.
# The actual text of the document. We display both the preprocessed and original versions of each document for comparison.

def search(query_text, k=5):
    #preprocess the query text
    preprocessed_query=preprocess_text(query_text)

    # generating query vector
    query_vector=embed_text([preprocessed_query])

    # performing the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

query_text='motorcycle'
distances, indices = search(query_text)

# Display the results
for i, idx in enumerate(indices[0]):
    # displaying preprocessed doc
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{processed_documents[idx]}\n")
