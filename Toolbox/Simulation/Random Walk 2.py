# Random Walk Simulation (1D, 2D, 3D) with Bias

import numpy as np
import matplotlib.pyplot as plt

# Random Walk Simulator Function
def random_walk(n_steps=1000, dim=1, bias=None):
    """
    Simulate a single random walk.
    n_steps : number of steps
    dim     : dimension (1, 2, or 3)
    bias    : array-like bias vector in [-1, 1] range; higher magnitude = stronger drift
              e.g., bias=0.5 (1D), [0.2, -0.3] (2D), [0.3, 0, -0.5] (3D)
    """
    if bias is None:
        bias = np.zeros(dim)
    bias = np.array(bias)

    # Steps: ±1 in each axis
    steps = np.random.choice([-1, 1], size=(n_steps, dim))
    
    # Apply bias — shift probability towards positive or negative directions
    for i in range(dim):
        prob_pos = 0.5 * (1 + np.clip(bias[i], -0.99, 0.99))
        steps[:, i] = np.where(np.random.rand(n_steps) < prob_pos, 1, -1)

    # Cumulative sum gives position
    positions = np.cumsum(steps, axis=0)
    return positions

# Multiple Runs 
def simulate_many_runs(n_runs=100, n_steps=1000, dim=1, bias=None):
    all_positions = np.zeros((n_runs, n_steps, dim))
    for i in range(n_runs):
        all_positions[i] = random_walk(n_steps, dim, bias)
    return all_positions

# MSD (Mean Squared Displacement) 
def compute_msd(all_positions):
    squared_displacements = np.sum(all_positions**2, axis=2)
    msd = np.mean(squared_displacements, axis=0)
    return msd

# Plot Functions 
def plot_average_displacement(all_positions, dim):
    mean_traj = np.mean(all_positions, axis=0)
    plt.figure(figsize=(6, 4))
    if dim == 1:
        plt.plot(mean_traj, label='Average Displacement (1D)')
        plt.xlabel('Steps'); plt.ylabel('Position')
    elif dim == 2:
        plt.plot(mean_traj[:, 0], mean_traj[:, 1], label='Average Path (2D)')
        plt.xlabel('X'); plt.ylabel('Y')
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], label='Average Path (3D)')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.legend()
    plt.title(f'Average Displacement in {dim}D')
    plt.show()

def plot_msd(msd):
    plt.figure(figsize=(6, 4))
    plt.plot(msd, label='Mean Squared Displacement')
    plt.xlabel('Steps'); plt.ylabel('MSD')
    plt.title('Mean Squared Displacement over Time')
    plt.legend()
    plt.show()

# Main module
def main():
    print("Random Walk Simulation")
    dim = int(input("Enter dimension (1, 2, or 3): "))
    n_steps = int(input("Enter number of steps (e.g., 1000): "))
    n_runs = int(input("Enter number of runs (e.g., 100): "))

    if dim == 1:
        bias = float(input("Enter bias scalar in [-1, 1]: "))
        bias_vec = [bias]
    elif dim == 2:
        bx = float(input("Enter bias for X axis [-1, 1]: "))
        by = float(input("Enter bias for Y axis [-1, 1]: "))
        bias_vec = [bx, by]
    elif dim == 3:
        bx = float(input("Enter bias for X axis [-1, 1]: "))
        by = float(input("Enter bias for Y axis [-1, 1]: "))
        bz = float(input("Enter bias for Z axis [-1, 1]: "))
        bias_vec = [bx, by, bz]
    else:
        raise ValueError("Dimension must be 1, 2, or 3")

    # Run simulations
    print(f"\nRunning {n_runs} random walks in {dim}D with bias = {bias_vec} ...")
    all_pos = simulate_many_runs(n_runs=n_runs, n_steps=n_steps, dim=dim, bias=bias_vec)
    
    # Compute and plot results
    plot_average_displacement(all_pos, dim)
    msd = compute_msd(all_pos)
    plot_msd(msd)

# Run 
if __name__ == "__main__":
    main()
