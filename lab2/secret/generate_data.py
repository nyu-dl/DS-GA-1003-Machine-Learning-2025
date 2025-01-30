# Usage: python lab2/secret/generate_data.py
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

OPTION = "svm_vs_logistic_regression"  # "chihuahua_muffin"  # "margin_classifier_alternate"  # "svm_vs_logistic_regression_alternate"
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
    X_class_0 = np.random.randn(10, 2) * 0.5 + [2.5, 2.5]  # Cluster 1
    X_class_1 = np.random.randn(10, 2) * 0.5 + [3.5, 3.5]  # Cluster 2

    # Add points close to the decision boundary
    X_close = np.array([[3.0, 3.1], [3.0, 2.9], [3.5, 3.5], [3.5, 3.6]])
    y_close = np.array([0, 0, 1, 1])  # Labels for close points

    # Combine the datasets
    X = np.vstack((X_class_0, X_class_1, X_close))
    y = np.hstack((np.zeros(len(X_class_0)), np.ones(len(X_class_1)), y_close))

    return X, y


def data_for_chihuahua_muffin():
    np.random.seed(42)

    # Generate two linearly separable clusters
    X_class_0 = np.random.randn(100, 2) * 0.5 + [3, 3]  # Cluster 1
    X_class_1 = np.random.randn(100, 2) * 0.5 + [3, 3]  # Cluster 2

    # Combine the datasets
    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack((np.zeros(len(X_class_0)), np.ones(len(X_class_1))))
    return X, y


def data_for_svm_vs_logistic_regression(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    # Generate two overlapping Gaussian clusters
    mean_class_0 = [1, 1]
    mean_class_1 = [4, 4]
    cov = [[1, 0.3], [0.3, 1]]  # Slight correlation between x and y

    X_class_0 = np.random.multivariate_normal(mean_class_0, cov, n_samples // 2)
    X_class_1 = np.random.multivariate_normal(mean_class_1, cov, n_samples // 2)

    # Merge the datasets
    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack(([0] * (n_samples // 2), [1] * (n_samples // 2)))

    # Introduce probabilistic label flipping near the boundary
    distances = np.abs(X[:, 0] - X[:, 1])  # Distance from the x=y diagonal
    flip_prob = np.exp(-distances)  # Flip probability decays with distance
    flip_mask = np.random.rand(n_samples) < flip_prob * 0.3  # Max 30% flip chance

    # Introduce probabilistic flipping of labels near the boundary
    y[flip_mask] = 1 - y[flip_mask]  # Flip 0 to 1 or 1 to 0

    return X, y


# Generate data
if "overlapped" in OPTION:
    X, y = make_blobs(n_samples=10, centers=2, cluster_std=4, center_box=(-5, 5), random_state=RANDOM_SEED)
elif "chihuahua_muffin" in OPTION:
    X, y = data_for_chihuahua_muffin()
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
if "chihuahua_muffin" in OPTION:
    df["label"] = df["y"].map({0: "chihuahua", 1: "muffin"})
df.to_csv(os.path.join(DATA_DIR, f"{OPTION}.csv"), index=False)

# Plot the data and save as png
import seaborn as sns

if "chihuahua_muffin" in OPTION:
    sns.scatterplot(x="x1", y="x2", hue="label", data=df)
else:
    sns.scatterplot(x="x1", y="x2", hue="y", data=df)
plt.savefig(os.path.join(DATA_DIR, f"{OPTION}.png"))
