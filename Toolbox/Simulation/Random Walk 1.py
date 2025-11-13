# Random Walk Simulation in 1D, 2D, 3D with Bias
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(101)

# Module 1 - 1D Random Walk (with bias)
def random_walk_1d(steps=1000, bias=0.0):
    """Simulate 1D random walk with bias (positive = drift to right).
        Default bias = 0.0 means P(+1) = P(-1) = 0.5"""
    moves = np.random.choice([-1, 1], size=steps)
    biased_moves = moves + bias  # add bias
    position = np.cumsum(biased_moves)
    return position

steps = 1000
bias = 0.1  # drift to the right
walk_1d = random_walk_1d(steps, bias)

plt.figure(figsize=(8,4))
plt.plot(walk_1d, color='blue')
plt.title(f'1D Random Walk (bias={bias})')
plt.xlabel('Step')
plt.ylabel('Position')
plt.grid(True)
plt.show()

# Module 2 - 2D Random Walk
def random_walk_2d(steps=1000, bias=(0.0, 0.0)):
    """Simulate 2D random walk with bias = (bias_x, bias_y)."""
    directions = np.random.choice(['up','down','left','right'], size=steps)
    x, y = np.zeros(steps), np.zeros(steps)
    for i, d in enumerate(directions):
        if d == 'up':
            x[i], y[i] = 0, 1
        elif d == 'down':
            x[i], y[i] = 0, -1
        elif d == 'left':
            x[i], y[i] = -1, 0
        else:
            x[i], y[i] = 1, 0
    x += bias[0]
    y += bias[1]
    pos_x, pos_y = np.cumsum(x), np.cumsum(y)
    return pos_x, pos_y

# Run and visualize 2D walk
steps = 100
bias = (0.2, -0.15)  # biases in x & y directions
x2d, y2d = random_walk_2d(steps, bias)

plt.figure(figsize=(6,6))
plt.plot(x2d, y2d, color='blue')
plt.scatter(x2d[0], y2d[0], color='red', label='Start')
plt.scatter(x2d[-1], y2d[-1], color='black', label='End')
plt.title(f'2D Random Walk (bias={bias})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Module 3 - 3D Random Walk
def random_walk_3d(steps=1000, bias=(0.0, 0.0, 0.0)):
    """Simulate 3D random walk with bias = (bx, by, bz)."""
    directions = np.random.choice(['x+','x-','y+','y-','z+','z-'], size=steps)
    dx, dy, dz = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    for i, d in enumerate(directions):
        if d == 'x+':
            dx[i] = 1
        elif d == 'x-':
            dx[i] = -1
        elif d == 'y+':
            dy[i] = 1
        elif d == 'y-':
            dy[i] = -1
        elif d == 'z+':
            dz[i] = 1
        elif d == 'z-':
            dz[i] = -1
    dx += bias[0]
    dy += bias[1]
    dz += bias[2]
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    z = np.cumsum(dz)
    return x, y, z

steps = 100
bias = (-0.1, 0.15, 0.2)  # bias toward -x,+y,+z
x3d, y3d, z3d = random_walk_3d(steps, bias)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x3d, y3d, z3d, color='blue')
ax.scatter(x3d[0], y3d[0], z3d[0], color='red', label='Start')
ax.scatter(x3d[-1], y3d[-1], z3d[-1], color='black', label='End')
ax.set_title(f'3D Random Walk (bias={bias})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
