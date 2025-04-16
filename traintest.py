import numpy as np
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# === 1. Load Train & Test Dataset ===
train_graphs = torch.load('train_graphs.pt')  # Dữ liệu đã cân bằng bằng SMOTE
test_graphs = torch.load('test_graphs.pt')    # Dữ liệu test giữ nguyên

# === 2. Chuyển dữ liệu PyG thành NumPy arrays ===
def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())  # Chuyển tensor thành numpy
        labels.append(data.y.item())  # Nhãn
    return np.array(features), np.array(labels)

X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)

# === 3. Định nghĩa Custom Decision Tree & Random Forest ===
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (feature_idx, threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature_idx, threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        def gini(y):
            counts = np.bincount(y)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        parent_impurity = gini(parent)
        left_impurity = gini(left_child)
        right_impurity = gini(right_child)

        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)

        return parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left_subtree, right_subtree = tree
        return self._predict_one(x, left_subtree if x[feature_idx] < threshold else right_subtree)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])


# === 4. Train Custom Random Forest ===
print("Training Custom Random Forest...")
rf = RandomForest(n_trees=10, max_depth=10, min_samples_split=5)
rf.fit(X_train, y_train)

# === 5. Evaluate Model ===
y_pred = rf.predict(X_test)

print("\n📌 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
