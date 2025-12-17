import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        # Layer 1: Graph Convolution
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Layer 2: Graph Convolution
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features (from Qwen), edge_index: Graph structure
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)