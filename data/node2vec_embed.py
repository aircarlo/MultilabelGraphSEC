import numpy as np
from tqdm import tqdm
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec


# load ground truth labels
gts = np.load('file_gts.npy')
# build adjacency matrix
C = gts.T @ gts
# C = C / np.diag(C)  # co-occurrence probability
np.fill_diagonal(C, 0)  # remove self-loops

adj = SparseTensor.from_dense(torch.from_numpy(C).float())
row, col, edge_attr = adj.t().coo()
edge_index = torch.stack([row, col], dim=0)

# G = Data(edge_index=edge_index, edge_attr=edge_attr, batch=torch.ones(200))  # include edge attributes
G = Data(edge_index=edge_index, batch=torch.ones(200))  # do not include edge attributes


def n2v_embed(model, loader, epochs):
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    loss_list = []
    for _ in tqdm(range(epochs)):
        loss = train()
        loss_list.append(loss)

    return loss_list


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Start embedding with {device}')
    embedding_dim = 64

    model = Node2Vec(G.edge_index,
                     embedding_dim=embedding_dim,
                     walk_length=20,
                     context_size=10,
                     walks_per_node=20,
                     p=0.1,
                     q=1,
                     sparse=True).to(device)
    loader = model.loader(batch_size=embedding_dim, shuffle=True, num_workers=8)
    loss_result = n2v_embed(model, loader, 250)

    # retrieve embedding with a forward call
    embs = model()

    # Save embedding
    torch.save(embs, 'n2v_64_embeddings.pth')
