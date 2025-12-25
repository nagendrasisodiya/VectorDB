import math

import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer


documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]


model=SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings=model.encode(documents)

print(embeddings.shape)
print(embeddings)

def euclidean_distance_fn(vector1, vector2):
    squared_sum=sum((x-y)**2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

euclidean_distance_fn(embeddings[0], embeddings[1])

l2_distance_manually=np.zeros([4, 4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j>i :
            l2_distance_manually[i,j]=euclidean_distance_fn(embeddings[i], embeddings[j])
        elif i>j :
            l2_distance_manually[i, j]=l2_distance_manually[j, i]

print("manually: ", l2_distance_manually)
# print(l2_distance_manually[0, 1])

# instead of calculating l2 distance manually we can also use scipy lib
l2_distance_scipy=scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print("scipy lib: ", l2_distance_scipy)

np.allclose(l2_distance_manually, l2_distance_scipy)
# print("manually: ", l2_distance_manually)
# print("scipy lib: ", l2_distance_scipy)


