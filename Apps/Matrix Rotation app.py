import tkinter as tk
from tkinter import StringVar
import numpy as np

# Default B_bar matrix (user can change this in the App GUI)
B_bar = np.array([[0.2 , -0.1],
                  [-0.1,  0.2]])
 
def update_matrix(theta_deg):
    """Updates the displayed matrix B based on theta."""
    theta = np.radians(float(theta_deg))
    C = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    B = np.dot(C, B_bar)
    
    for i in range(2):
        for j in range(2):
            B_labels[i][j].set(f"{B[i, j]:.4f}")

def update_B_bar():
    """Updates B_bar based on user input."""
    global B_bar
    try:
        B_bar = np.array([[float(entry_B_bar[0][0].get()), float(entry_B_bar[0][1].get())],
                           [float(entry_B_bar[1][0].get()), float(entry_B_bar[1][1].get())]])
        update_matrix(theta_slider.get())
    except ValueError:
        pass

# Create GUI window
root = tk.Tk()
root.title("Matrix Rotation Interface")
root.geometry("750x400")  # App window size

tk.Label(root, text="Enter 2x2 Matrix B_bar:").grid(row=0, column=0, columnspan=2)

entry_B_bar = [[None, None], [None, None]]
for i in range(2):
    for j in range(2):
        entry_B_bar[i][j] = tk.Entry(root, width=10)
        entry_B_bar[i][j].grid(row=i+1, column=j, padx=5, pady=5)
        entry_B_bar[i][j].insert(0, str(B_bar[i, j]))

tk.Button(root, text="Update B_bar", command=update_B_bar).grid(row=3, column=0, columnspan=2, pady=5)

tk.Label(root, text="Select \u03B8 (degrees):").grid(row=4, column=0, pady=5) # Can also copy-paste θ directly
theta_slider = tk.Scale(root, from_=0, to=360, orient=tk.HORIZONTAL, length=600, resolution=0.5, command=update_matrix)
theta_slider.grid(row=4, column=1, pady=5)
theta_slider.set(0)

tk.Label(root, text="Computed Matrix B:").grid(row=5, column=0, columnspan=2, pady=5)
B_labels = [[StringVar(), StringVar()], [StringVar(), StringVar()]]
for i in range(2):
    for j in range(2):
        label = tk.Label(root, textvariable=B_labels[i][j], width=12, relief="sunken", padx=5, pady=5)
        label.grid(row=i+6, column=j, padx=5, pady=5)

update_matrix(0)  # Initialize B with θ = 0
root.mainloop()
