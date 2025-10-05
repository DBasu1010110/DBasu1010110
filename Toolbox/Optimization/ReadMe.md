Optimization algorithms (like GDA) only make sense when applied on a defined objective (or cost) function to minimize or maximize.
Otherwise, you’re just iterating numbers with no purpose.

Even though Iris is primarily used for classification, we repurpose it slightly to define an optimization problem:
Train a logistic regression classifier to predict whether a flower is Iris Setosa (1) or not (0).
The goal is then to the logistic loss function.

Why is it good?
> It’s smooth and convex.
> Gradient-based methods perform differently depending on learning rate and other parameters.
> We can visualize convergence easily.

Objective Function: binary cross-entropy loss (c.f.: <https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning>)

Program design: (use the comments #------------------ <number>) to break the program into modules for clarity and ease of editing
1. Import libraries and load the dataset
2. Helper Functions
3. GDA implementations
4. Compare the methods (using loss)
5. Evaluate on Test set

**Note**: All the methods can be directly applied using pre-built functions in sklearn (vanilla, stochastic, mini-batch) and pyTorch (momentum, adam)
