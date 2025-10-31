Discussion:

Gradient-free methods are useful when gradients are noisy, unavailable, or discontinuous.

1. Nelder-Mead, Powell, COBYLA: Fast local optimizers for small problems.
2. Genetic Algorithm: Slower, but can escape local minima.
3. For larger problems, use:
  - scipy.optimize.dual_annealing (Simulated Annealing)
  - nevergrad / DEAP / optuna (Evolutionary strategies)
  - cma / PyGMO (CMA-ES, global DFO)
