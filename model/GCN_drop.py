import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import random
# === 1. Đặt seed để đảm bảo tính nhất quán ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

# === 3. Load dữ liệu ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load('graphs.pt',weights_only=False)
print("Số lượng đồ thị:", len(graphs))

graphs_0 = [g for g in graphs if g.y.item() == 0]
graphs_1 = [g for g in graphs if g.y.item() == 1]
min_size = min(len(graphs_0), len(graphs_1))
print(f"Cân bằng tập dữ liệu về {min_size} mẫu mỗi lớp...")

balanced_graphs = graphs_0[:min_size] + graphs_1[:min_size]
remaining_graphs = graphs_0[min_size:] + graphs_1[min_size:]
random.shuffle(balanced_graphs)
random.shuffle(remaining_graphs)

# === 4. Chia Train/Test ===
print("Chia tập train (80%) và test (20%)...")
train_graphs, test_graphs = train_test_split(
    balanced_graphs, test_size=0.2, random_state=42, stratify=[g.y.item() for g in balanced_graphs]
)

# Thêm dữ liệu dư vào tập train và test
train_graphs += remaining_graphs[:18000]
test_graphs += remaining_graphs[18000:]

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

X_train, y_train = extract_features(train_graphs)
X_test, y_test = extract_features(test_graphs)

# === 6. Cân bằng dữ liệu bằng Dropout ===
def apply_dropout(features, labels, dropout_rate=0.2, num_samples=2):
    balanced_features, balanced_labels = [], []
    
    for i in range(len(features)):
        if labels[i] == 1:  # Chỉ áp dụng cho lớp thiểu số
            for _ in range(num_samples):
                mask = np.random.binomial(1, 1 - dropout_rate, size=features.shape[1])
                new_sample = features[i] * mask
                balanced_features.append(new_sample)
                balanced_labels.append(labels[i])
    
    balanced_features = np.vstack(balanced_features)
    balanced_labels = np.hstack(balanced_labels)
    
    features = np.vstack([features, balanced_features])
    labels = np.hstack([labels, balanced_labels])
    
    return features, labels

X_train_balanced, y_train_balanced = apply_dropout(X_train, y_train, dropout_rate=0.0001, num_samples=10)
print("Trước Dropout:", np.bincount(y_train))
print("Sau Dropout:", np.bincount(y_train_balanced))

# === 7. Chuyển lại thành dữ liệu đồ thị PyG ===
def convert_to_graphs(X, y):
    new_graphs = []
    for i in range(len(X)):
        graph_data = Data(
            x=torch.tensor(X[i], dtype=torch.float).unsqueeze(0),
            y=torch.tensor([y[i]], dtype=torch.long)
        )
        new_graphs.append(graph_data)
    return new_graphs

train_graphs_balanced = convert_to_graphs(X_train_balanced, y_train_balanced)
test_graphs_final = convert_to_graphs(X_test, y_test)

# === 8. Lưu dữ liệu sau khi xử lý ===
torch.save(train_graphs_balanced, 'train_graphs.pt')
torch.save(test_graphs_final, 'test_graphs.pt')
print("Đã lưu train_graphs.pt và test_graphs.pt.")
