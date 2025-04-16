import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import Linear

# === 1. Define the Model ===
class NodeGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.sigmoid(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EdgeCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, edge_attr):
        if edge_attr is None:
            return None
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.permute(1, 0).unsqueeze(0)
        edge_attr = F.sigmoid(self.conv1(edge_attr))
        edge_attr = self.conv2(edge_attr)
        return edge_attr.squeeze(0).permute(1, 0)

class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_output_dim, final_dim):
        super(GraphModel, self).__init__()
        self.node_gcn = NodeGraphSAGE(node_input_dim, node_hidden_dim, node_output_dim)
        self.edge_cnn = EdgeCNN(edge_input_dim, edge_output_dim)
        self.att = GlobalAttention(gate_nn=Linear(node_output_dim, 1))
        graph_feature_dim = node_output_dim + edge_output_dim
        self.fc = nn.Linear(graph_feature_dim, final_dim)

    def forward(self, data):
        node_features = self.node_gcn(data.x, data.edge_index)
        edge_features = self.edge_cnn(data.edge_attr) if data.edge_attr is not None else None
        node_features = self.att(node_features, data.batch)
        if edge_features is not None:
            edge_batch = data.batch[data.edge_index[0]]
            edge_features = global_mean_pool(edge_features, edge_batch)
            graph_features = torch.cat([node_features, edge_features], dim=-1)
        else:
            graph_features = node_features
        out = self.fc(graph_features)
        return out, graph_features

# === 1. Load Data ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load('graphs.pt',weights_only=False)


print("Length cua graphs: ",len(graphs))

graphs_0 = [g for g in graphs if g.y.item() == 0]
graphs_1 = [g for g in graphs if g.y.item() == 1]

# Lấy số lượng nhỏ hơn giữa hai loại nhãn
min_size = min(len(graphs_0), len(graphs_1))
print(f"Balancing dataset to {min_size} samples per class...")


balanced_graphs = graphs_0[:min_size] + graphs_1[:min_size]
remaining_graphs = graphs_0[min_size:] + graphs_1[min_size:]
np.random.shuffle(balanced_graphs)
np.random.shuffle(remaining_graphs)

# === 4. Split into Train/Test Sets ===
print("Splitting into train (80%) and test (20%)...")
train_graphs, test_graphs = train_test_split(
    balanced_graphs, test_size=0.2, random_state=11, stratify=[g.y.item() for g in balanced_graphs]
)

size_remaining_graphs=len(remaining_graphs)
print("Do dai cua data: ",len(balanced_graphs))
print("Length cua data re con lai: ",len(remaining_graphs))
train_graphs=train_graphs+remaining_graphs[:18000]
test_graphs = test_graphs+remaining_graphs[18000:]
# === 2. Extract Features from GraphModel ===
model = GraphModel(node_input_dim=100, node_hidden_dim=64, node_output_dim=32,
                   edge_input_dim=50, edge_output_dim=16, final_dim=2).to(device)
model.eval()

def extract_features(graphs):
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    features, labels = [], []
    print(len(loader))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, graph_features = model(data)
            features.append(graph_features.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y

# Trích xuất đặc trưng cho tập train và test
X_train, y_train = extract_features(train_graphs)
X_test, y_test = extract_features(test_graphs)

# === 3. Apply SMOTE to Train Set ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_resampled))

# === 4. Convert Back to PyG Data ===
def convert_to_graphs(X, y):
    new_graphs = []
    for i in range(len(X)):
        graph_data = Data(
            x=torch.tensor(X[i], dtype=torch.float).unsqueeze(0),
            y=torch.tensor([y[i]], dtype=torch.long)
        )
        new_graphs.append(graph_data)
    return new_graphs

# Tạo tập train và test mới
train_graphs_resampled = convert_to_graphs(X_train_resampled, y_train_resampled)
test_graphs_final = convert_to_graphs(X_test, y_test)

# === 5. Save Processed Data ===
torch.save(train_graphs_resampled, 'train_graphs.pt')
torch.save(test_graphs_final, 'test_graphs.pt')
print("Saved train_graphs.pt (balanced) and test_graphs.pt (original).")