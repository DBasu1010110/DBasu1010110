# Derivative Free Optimization

# Module 1 - import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Module 2 - Data prep and setup
data = load_diabetes()
X, y = data.data[:, :3], data.target   # using first 3 features for simplicity
y = y.reshape(-1, 1)

# Truncate dataset to 100 samples for speed
X, y = X[:100], y[:100]

# Tain/Test split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=86)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add an intercept
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

n_features = X_train.shape[1]
theta_init = np.zeros(n_features)

# Module 3 - Define Cost Function (using MSE)

def cost_function(theta, X, y):
    """Mean Squared Error"""
    y_pred = X @ theta
    return np.mean((y - y_pred)**2)

# Module 4 - In-bult DFOs (Scipy)
# Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# 4.1 Nelder-Mead
res_nm = minimize(cost_function, theta_init, args=(X_train, y_train),
                  method='Nelder-Mead', options={'maxiter': 500, 'disp': False})

# 4.2 Powell
res_pw = minimize(cost_function, theta_init, args=(X_train, y_train),
                  method='Powell', options={'maxiter': 500, 'disp': False})

# 4.3 COBYLA (no constraints, but can pass dummy empty list)
res_cb = minimize(cost_function, theta_init, args=(X_train, y_train),
                  method='COBYLA', constraints=[], options={'maxiter': 500, 'disp': False})

# Module 5 - Genetic Algorithm (a simplified implementation)

def genetic_algorithm(X, y, pop_size=20, generations=60, mutation_rate=0.15):
    np.random.seed(0)
    n_features = X.shape[1]
    population = np.random.randn(pop_size, n_features)
    best_losses = []

    for g in range(generations):
        losses = np.array([cost_function(ind, X, y) for ind in population])
        fitness = 1 / (1 + losses)
        top_idx = np.argsort(losses)[:int(0.3 * pop_size)]
        parents = population[top_idx]

        # Crossover
        children = []
        while len(children) < pop_size - len(parents):
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            cross = np.random.randint(1, n_features)
            child = np.concatenate((p1[:cross], p2[cross:]))
            children.append(child)

        # Mutation
        children = np.array(children)
        mutations = np.random.rand(*children.shape) < mutation_rate
        children[mutations] += np.random.randn(np.sum(mutations))

        population = np.vstack((parents, children))
        best_losses.append(losses[top_idx[0]])

    best_idx = np.argmin([cost_function(ind, X, y) for ind in population])
    return population[best_idx], best_losses

theta_ga, ga_losses = genetic_algorithm(X_train, y_train)

# Genetic algorithm convergence plot

plt.figure(figsize=(7,5))
plt.plot(ga_losses, color='teal', linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Best Loss (MSE)")
plt.title("Genetic Algorithm Convergence")
plt.grid(True)
plt.show()

# Module 6 - DFO Results (comparison)

methods = {
    "Nelder-Mead": res_nm,
    "Powell": res_pw,
    "COBYLA": res_cb,
    "Genetic Algorithm": theta_ga
}

print("DFO Comparison (Train/Test MSE):\n")
for name, res in methods.items():
    theta = res.x if hasattr(res, 'x') else res
    train_mse = cost_function(theta, X_train, y_train)
    test_mse = cost_function(theta, X_test, y_test)
    print(f"{name:<20} | Train MSE: {train_mse:>10.4f} | Test MSE: {test_mse:>10.4f}")

