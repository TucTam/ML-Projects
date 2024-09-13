import random
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Returns the distances of the pivot and all the other points
def all_distances(pivot, dataset):
    distances = []
    for x in dataset:
        distances.extend([np.linalg.norm(x-pivot)])
    return distances

# Returns the index of location that contains the nth smallest value
def find_nth_smallest_index(container, n):
    temp = sorted(container)
    return np.where(container == temp[n])[0][0]
    
# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        terms = input_term.split("-")
        z = W[vocab[terms[2]]] + (W[vocab[terms[1]]] - W[vocab[terms[0]]])
        distances = all_distances(z, W)
        
        a = [find_nth_smallest_index(distances, 0), find_nth_smallest_index(distances, 1),]
        
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for x in a:
            print("%35s\t\t%f\n" % (ivocab[x], distances[x]))