import torch
import numpy as np
from model.GCNModel import GCNModel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import json

# Load dữ liệu
graphs = torch.load('graphs.pt')

print(graphs[0].x)
print(graphs[0].y)

model = GCNModel(in_channels=100, hidden_channels=64, out_channels=10)


features = []
labels = []


for idx, data in enumerate(graphs):
    out_features = model(data)
    features.append(np.mean(out_features.detach().numpy(), axis=0)) 
    labels.append(data.y.detach().numpy()[0])  


X = np.array(features, dtype=np.float32)
X = np.nan_to_num(X, nan=0) 
y = np.array(labels)


X = X.reshape(X.shape[0], -1)  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


from collections import Counter
train_label_counts = Counter(y_train_resampled)
print(f"Số lượng nhãn trong train sau SMOTE: {train_label_counts}")


# Tạo dữ liệu dạng danh sách các dictionary
train_data = [
    {"features": feature, "label": label}
    for feature, label in zip(X_train_resampled.tolist(), y_train_resampled.tolist())
]

valid_data = [
    {"features": feature, "label": label}
    for feature, label in zip(X_valid.tolist(), y_valid.tolist())
]

test_data = [
    {"features": feature, "label": label}
    for feature, label in zip(X_test.tolist(), y_test.tolist())
]

# Lưu dữ liệu vào các file JSON riêng biệt
with open("train.json", "w") as train_file:
    json.dump(train_data, train_file, indent=4)  # indent=4 để dễ đọc hơn

with open("valid.json", "w") as valid_file:
    json.dump(valid_data, valid_file, indent=4)

with open("test.json", "w") as test_file:
    json.dump(test_data, test_file, indent=4)

print("Dữ liệu đã được lưu vào các file 'train.json', 'valid.json', và 'test.json'")

