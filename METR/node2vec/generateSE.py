import pickle

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

import node2vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../data/PEMS04.csv'
SE_file = '../data/PEMS04-SE.txt'
node_num = 307
# id_filename = "../data/PEMS03-id.txt"
id_filename = None

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])

            if id_filename:
                i = id_dict[i]
                j = id_dict[j]
            A[i, j] = 1
            A[j, i] = 1
            distaneA[i, j] = distance
            distaneA[j, i] = distance
    return A, distaneA


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=iter)
    model.wv.save_word2vec_format(output_file)

    return


adj_np, _ = get_adjacency_matrix_2direction(Adj_file, node_num, id_filename=id_filename)
output = open('PEMS04_adj_mx.pkl', 'wb')
pickle.dump(adj_np, output)
output.close()
nx_G = nx.from_numpy_array(A=adj_np, create_using=nx.DiGraph())
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
