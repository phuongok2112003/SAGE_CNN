import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# === 1. Load Train & Test Dataset ===
train_graphs = torch.load('train_graphs.pt')  # Đã cân bằng bằng SMOTE
test_graphs = torch.load('test_graphs.pt')    # Giữ nguyên, không SMOTE

# Chuyển đổi dữ liệu PyG thành numpy array
def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())  # Chuyển tensor thành numpy
        labels.append(data.y.item())
    return np.array(features), np.array(labels)

# Chuyển train và test thành numpy
X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)

# === 2. Cân bằng dữ liệu bằng SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 3. Huấn luyện mô hình XGBoost với GridSearch ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_

# === 4. Đánh giá mô hình ===
y_pred = best_model.predict(X_test)

# === 5. Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (TP + FP + FN)

# In kết quả TP, FP, FN, TN
for i, label in enumerate(np.unique(y_test)):
    print(f"\nClass {label}:")
    print(f"TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")

# === 6. In đánh giá mô hình ===
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7. Vẽ Confusion Matrix ===
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
