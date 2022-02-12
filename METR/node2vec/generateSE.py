import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../data/Adj(METR).txt'
SE_file = '../data/SE(METR).txt'


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices):
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
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


adj_np, _ = get_adjacency_matrix_2direction(Adj_file, node_num)
# output = open('PEMS08_adj_mx.pkl', 'wb')
# pickle.dump(adj_np, output)
# output.close()
nx_G = nx.from_numpy_array(adj_np, nx.DiGraph())
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)