import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, GATConv
import torch.nn.functional as F
from net_models.crnn_net import CRNN


class GGNN(torch.nn.Module):  # Gated Graph Neural Network model
    def __init__(self, input_dim, hidden_layers):
        super(GGNN, self).__init__()
        self.conv = GatedGraphConv(out_channels=input_dim, num_layers=hidden_layers, aggr='add', bias=True)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index, data.edge_attr)
        return x


class GATNN(torch.nn.Module):  # Graph Attention Network model
    def __init__(self, in_channels, out_channels):
        super(GATNN, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


class CRNN_GGNN(nn.Module):
    """CRNN-GNN aggregation"""

    def __init__(self, emb_dim):
        super(CRNN_GGNN, self).__init__()
        self.crnn = CRNN(out_dim=emb_dim)
        self.ggnn = GGNN(input_dim=emb_dim, hidden_layers=3)  # Gated Graph Neural Network

    def forward(self, spec_data, graph_data):
        spec_out = self.crnn(spec_data, None)
        graph_out = self.ggnn(graph_data)
        out = torch.matmul(spec_out, graph_out.T)
        return out
