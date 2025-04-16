import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # Batch Normalization
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x_res = x  # Lưu lại đầu vào ban đầu

        x = self.gcn1(x, edge_index)
        x = self.bn1(x)  # Batch Normalization
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gcn3(x, edge_index)

        # Residual Connection (nếu số chiều không khớp, cần một linear projection)
        if x_res.shape[1] == x.shape[1]:
            x = x + x_res  # Residual Connection

        return x
