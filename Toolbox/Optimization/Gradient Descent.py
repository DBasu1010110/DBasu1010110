# Gradient Descent Algorithms (GDA)
# Dataset: Iris (Binary Classification: Setosa vs Others) using a cost function

# ------------------------ 1 
# Imports and Data Setup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]   # extract sepal length and petal length
y = (iris.target == 0).astype(int)  # 1 if Setosa, 0 otherwise

# Train/Test split and standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=86)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize parameters
np.random.seed(86)
theta_init = np.random.randn(X_train.shape[1])

# ------------------------ 2 
# Helper Functions

def sigmoid(z): #Sigmoid activation function
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, theta): #Binary cross-entropy loss
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-7  # small value to avoid log(0)
    return - (1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

def compute_gradient(X, y, theta): # Gradient of the loss with respect to theta
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * X.T @ (h - y)

# ------------------------ 3 
# Call functions for GD Algorithms

def gradient_descent(X, y, theta, lr=0.01, epochs=250):
    losses = []
    for i in range(epochs):
        grad = compute_gradient(X, y, theta)
        theta -= lr * grad
        losses.append(compute_loss(X, y, theta))
    return theta, losses

def stochastic_gradient_descent(X, y, theta, lr=0.01, epochs=250):
    m = len(y)
    losses = []
    for epoch in range(epochs):
        for i in range(m):
            rand_idx = np.random.randint(0, m)
            Xi = X[rand_idx:rand_idx+1]
            yi = y[rand_idx:rand_idx+1]
            grad = compute_gradient(Xi, yi, theta)
            theta -= lr * grad
        losses.append(compute_loss(X, y, theta))
    return theta, losses

def mini_batch_gradient_descent(X, y, theta, lr=0.01, epochs=250, batch_size=10):
    m = len(y)
    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            grad = compute_gradient(X_batch, y_batch, theta)
            theta -= lr * grad
        losses.append(compute_loss(X, y, theta))
    return theta, losses

def momentum_gradient_descent(X, y, theta, lr=0.01, beta=0.9, epochs=250):
    v = np.zeros_like(theta)
    losses = []
    for i in range(epochs):
        grad = compute_gradient(X, y, theta)
        v = beta * v + (1 - beta) * grad
        theta -= lr * v
        losses.append(compute_loss(X, y, theta))
    return theta, losses

def adam_optimizer(X, y, theta, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-7, epochs=250):
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    losses = []
    for t in range(1, epochs + 1):
        grad = compute_gradient(X, y, theta)
        m_t = beta1 * m_t + (1 - beta1) * grad
        v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(compute_loss(X, y, theta))
    return theta, losses

# ------------------------ 4 
# Compare the methods based on loss

methods = {
    "Gradient Descent": gradient_descent,
    "Stochastic GD": stochastic_gradient_descent,
    "Mini-batch GD": mini_batch_gradient_descent,
    "Momentum": momentum_gradient_descent,
    "Adam": adam_optimizer
}

results = {}

for name, func in methods.items():
    theta = theta_init.copy()
    theta_opt, losses = func(X_train, y_train, theta)
    results[name] = {"theta": theta_opt, "losses": losses}
    print(f"{name} ---> Final Loss: {losses[-1]:.4f}")


# ------------------------ 5 
# Evaluate on Test Set

def accuracy(X, y, theta):
    preds = sigmoid(X @ theta) >= 0.5
    return np.mean(preds == y)

for name, res in results.items():
    acc = accuracy(X_test, y_test, res["theta"])
    print(f"{name} ---> Test Accuracy: {acc:.4f}")


