# Usage: python lab3/secret/generate_data.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OPTION = "xor_pattern_label"  # "xor_pattern"
DATA_DIR = "lab3/secret/data"
RANDOM_SEED = 42
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Control random seed
np.random.seed(RANDOM_SEED)


def data_for_xor_pattern(n_samples=500):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR pattern
    return X, y


# Generate data
if "xor_pattern" in OPTION:
    X, y = data_for_xor_pattern()
else:
    raise ValueError(f"Invalid option: {OPTION}")
y = (y > 0).astype(int)  # Binary labels (0 and 1)

# Shuffle data
shuffle_indices = np.random.permutation(len(X))
X_random = X[shuffle_indices]
y_random = y[shuffle_indices]

X_ordered = X_random
y_ordered = y_random

# Save as pandas dataframe
df = pd.DataFrame(np.hstack([X_ordered, y_ordered.reshape(-1, 1)]), columns=["x1", "x2", "y"])
df.to_csv(os.path.join(DATA_DIR, f"{OPTION}.csv"), index=False)

# Plot the data and save as png
if "label" in OPTION:
    df["label"] = df["y"].map({0: "blueberry muffin", 1: "chihuahua"})
    sns.scatterplot(x="x1", y="x2", hue="label", data=df)
else:
    sns.scatterplot(x="x1", y="x2", hue="y", data=df)
save_path = os.path.join(DATA_DIR, f"{OPTION}.png")
plt.savefig(save_path)
print(f"Data plot saved to {save_path}")
