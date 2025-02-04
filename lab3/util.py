import matplotlib.pyplot as plt
import numpy as np
import torch


def forward_pass(X, W1, b1, W2, b2):
    """Manually compute MLP forward pass with ReLU activation."""
    z1 = X @ W1 + b1  # First layer
    a1 = torch.relu(z1)  # ReLU activation
    z2 = a1 @ W2 + b2  # Second layer (logits)
    p_hat = torch.softmax(z2, dim=1)  # Convert logits to probabilities
    return p_hat.argmax(dim=1)  # Get class predictions


def plot_decision_boundary(W1, b1, W2, b2, X, y, resolution=100):
    """
    Plots the decision boundary using the provided model weights.

    Parameters:
    - W1, b1, W2, b2: Trained weights & biases of the MLP
    - X: Training data (N, 2) for visualization
    - y: Training labels (N,)
    - resolution: Grid resolution (higher means finer boundary)
    """

    # Define grid range (slightly larger than X range)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Convert grid to tensor for model input
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Forward pass using only weights (no model class needed)
    with torch.no_grad():
        pred_classes = forward_pass(grid_tensor, W1, b1, W2, b2).numpy()

    # Reshape predictions to match grid shape
    Z = pred_classes.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of MLP")
    plt.show()


def plot_loss(loss_history_sgd, loss_history_adam):

    # =================== Plot Loss Curves ===================
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_history_sgd, label="SGD", linestyle="dashed")
    plt.plot(loss_history_adam, label="Adam", linestyle="solid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SGD vs. Adam: Training Loss")
    plt.legend()
    plt.show()
