import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# === 1. Load Train & Test Dataset ===
train_graphs = torch.load('train_graphs.pt', weights_only=False)
test_graphs = torch.load('test_graphs.pt', weights_only=False)

def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())
        labels.append(data.y.item())
    return np.array(features), np.array(labels)

X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# === 3. Định nghĩa Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

# === 4. Xây dựng MLP Model ===
class MLP(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
             nn.Linear(64, 2), nn.Softmax(dim=1) 
        )
    
    def forward(self, x):
        return self.layers(x)
    
input_size = 48
print(input_size)
loaded_model = MLP(input_size, dropout_rate=0.3)

# Load trọng số đã lưu
loaded_model.load_state_dict(torch.load("best_mlp_model.pth"))
loaded_model.eval()  # Chuyển sang chế độ đánh giá
print("Model loaded successfully!")

# === 10. Kiểm tra mô hình đã tải lại ===
with torch.no_grad():
    y_pred_tensor = loaded_model(X_test_tensor)
    y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()

# === 11. Đánh giá kết quả ===
conf_matrix = confusion_matrix(y_test_tensor, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_tensor, y_pred, average='macro')

print(f"\nTest Accuracy: {accuracy_score(y_test_tensor, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(classification_report(y_test_tensor, y_pred))

# === 12. Vẽ Confusion Matrix ===
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
