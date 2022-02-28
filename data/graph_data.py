import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def create_graph(embedding_file, verbose):
    gts = np.load('data\\file_gts.npy')
    print(f'load initial node embedding from {embedding_file}... ', end='')
    if embedding_file[-3:] == 'npy':
        x0 = np.load(embedding_file)
        x0 = torch.from_numpy(x0).float()
    elif embedding_file[-3:] == 'pth':
        x0 = torch.load(embedding_file)
    else:
        raise ValueError('Initial embedding not defined')
    print('done')

    print('building graph... ', end='')
    C = gts.T @ gts  # compute labels co-occurence binary matrix
    C = C / np.diag(C)  # compute labels co-occurrence probability matrix
    np.fill_diagonal(C, 0)  # remove self-loops

    adj = SparseTensor.from_dense(torch.from_numpy(C).float())
    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    G = Data(x=x0, edge_index=edge_index, edge_attr=edge_attr, batch=torch.ones(200))
    print('done\n')

    if verbose:
        print('Graph statistics:')
        print(G)
        print(f'Number of nodes: {G.num_nodes}')
        print(f'Number of edges: {G.num_edges}')
        print(f'Average node degree: {G.num_edges / G.num_nodes:.2f}')
        print(f'Has isolated nodes: {G.has_isolated_nodes()}')
        print(f'Has self-loops: {G.has_self_loops()}')
        print(f'Is undirected: {G.is_undirected()}\n')

    return G
