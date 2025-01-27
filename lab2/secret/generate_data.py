# Usage: python lab2/secret/generate_data.py
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

OPTION = "svm_vs_logistic_regression_alternate"  # "overlapped_alternate"  # "margin_classifier_alternate"  # "linearly_separable_alternate"  # "overlapped_alternate"  # "overlapped"  # "linearly_separable"
DATA_DIR = "lab2/secret/data"
RANDOM_SEED = 42
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# control random seed
np.random.seed(RANDOM_SEED)

import matplotlib.pyplot as plt
import numpy as np


def data_for_perceptron_vs_svm():
    np.random.seed(42)

    # Generate two linearly separable clusters
    X_class_0 = np.random.randn(10, 2) * 0.5 + [2, 2]  # Cluster 1
    X_class_1 = np.random.randn(10, 2) * 0.5 + [4, 4]  # Cluster 2

    # Add points close to the decision boundary
    X_close = np.array([[3.0, 3.1], [3.0, 2.9], [3.5, 3.5], [3.5, 3.6]])
    y_close = np.array([0, 0, 1, 1])  # Labels for close points

    # Combine the datasets
    X = np.vstack((X_class_0, X_class_1, X_close))
    y = np.hstack((np.zeros(len(X_class_0)), np.ones(len(X_class_1)), y_close))

    return X, y


def data_for_svm_vs_logistic_regression():
    # Generate two clusters
    X_class_0 = np.array([[1, 2], [2, 1], [1.5, 1.5], [1.5, 2]])  # Cluster 0
    X_class_1 = np.array([[3, 4], [4, 3], [3.5, 3.5], [3.5, 4]])  # Cluster 1

    # Add overlap near the boundary
    X_overlap = np.array([[2.5, 2.5], [2.6, 2.4]])
    y_overlap = np.array([0, 1])  # One point from each class

    # Combine the datasets
    X = np.vstack((X_class_0, X_overlap, X_class_1))
    y = np.hstack(([0] * len(X_class_0), [1] * len(X_class_1), y_overlap))
    return X, y


# Generate data
if "overlapped" in OPTION:
    X, y = make_blobs(n_samples=10, centers=2, cluster_std=4, center_box=(-5, 5), random_state=RANDOM_SEED)
elif "linearly_separable" in OPTION:
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
elif "margin_classifier" in OPTION:
    X, y = data_for_perceptron_vs_svm()
elif "svm_vs_logistic_regression" in OPTION:
    X, y = data_for_svm_vs_logistic_regression()
else:
    raise ValueError(f"Invalid option: {OPTION}")
y = (y > 0).astype(int)  # Binary labels (0 and 1)

# Sequential order (sorted by one feature)
X_sequential = X[np.argsort(X[:, 0])]
y_sequential = y[np.argsort(X[:, 0])]

# Random order
shuffle_indices = np.random.permutation(len(X))
X_random = X[shuffle_indices]
y_random = y[shuffle_indices]

if "alternate" in OPTION:
    # Alternate examples between classes
    X_ordered = np.vstack([X[y == 0][i // 2] if i % 2 == 0 else X[y == 1][i // 2] for i in range(len(X))])
    y_ordered = np.hstack([0 if i % 2 == 0 else 1 for i in range(len(X))])

else:
    X_ordered = X_random
    y_ordered = y_random

# Save as pandas dataframe
df = pd.DataFrame(np.hstack([X_ordered, y_ordered.reshape(-1, 1)]), columns=["x1", "x2", "y"])
df.to_csv(os.path.join(DATA_DIR, f"{OPTION}.csv"), index=False)

# Plot the data and save as png
import seaborn as sns

sns.scatterplot(x="x1", y="x2", hue="y", data=df)
plt.savefig(os.path.join(DATA_DIR, f"{OPTION}.png"))
