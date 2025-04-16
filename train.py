import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


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

input_size = X_train.shape[1]
model = MLP(input_size, dropout_rate=0.3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


num_epochs = 30
batch_size = 32
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    best_model_state = model.state_dict()


model_path = "best_mlp_model.pth"

if os.path.exists(model_path):
    old_model = MLP(input_size, dropout_rate=0.3)
    old_model.load_state_dict(torch.load(model_path))
    old_model.eval()

    with torch.no_grad():
        old_y_pred_tensor = old_model(X_test_tensor)
        old_y_pred = torch.argmax(old_y_pred_tensor, dim=1).numpy()

    old_precision, old_recall, old_f1, _ = precision_recall_fscore_support(y_test, old_y_pred, average='macro')
    print(f"Old Model - Precision: {old_precision:.4f}, Recall: {old_recall:.4f}, F1-score: {old_f1:.4f}")
else:
    old_precision, old_recall, old_f1 = 0, 0, 0

if precision > old_precision and recall > old_recall and f1 > old_f1:
    torch.save(best_model_state, model_path)
    print("New model is better in all aspects. Model saved successfully!")
else:
    print("Old model is better. Keeping previous model.")


loaded_model = MLP(input_size, dropout_rate=0.3)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

with torch.no_grad():
    y_pred_tensor = loaded_model(X_test_tensor)
    y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()


conf_matrix = confusion_matrix(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print(f"\nTest Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
