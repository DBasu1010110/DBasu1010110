'''
OLS Regression in 3 ways:
1. Direct computation (Normal Equation)
2. Using Python libraries (statsmodels)
3. Gradient Descent implementation
'''

'''
Modules for easy implementation
Y: Dependant Variable
X: Matrix of independent variables
beta: regression coefficients
'''

# MODULE 1: OLS using statsmodels
def ols_direct(X, Y):
    """
    Compute OLS coefficients using matrix algebra.
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    Y_hat = X @ beta
    residuals = Y - Y_hat
    r2 = 1 - np.sum(residuals**2) / np.sum((Y - np.mean(Y))**2)
    return {"method": "Direct", "beta": beta, "Y_hat": Y_hat, "residuals": residuals, "r2": r2}

# MODULE 2: OLS using statsmodels
def ols_library(X, Y):
    """
    Compute OLS coefficients using statsmodels.
    """
    model = sm.OLS(Y, X).fit()
    return {"method": "Statsmodels", "beta": model.params, "Y_hat": model.fittedvalues, "residuals": model.resid, "r2": model.rsquared}

# MODULE 3: OLS using Gradient Descent
def ols_gradient_descent(X, Y, lr=0.01, epochs=10000):
    """
    Compute OLS coefficients using gradient descent.
    """
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(epochs):
        gradient = -2/n * (X.T @ (Y - X @ beta))
        beta -= lr * gradient

    Y_hat = X @ beta
    residuals = Y - Y_hat
    r2 = 1 - np.sum(residuals**2) / np.sum((Y - np.mean(Y))**2)
    return {"method": "Gradient Descent", "beta": beta, "Y_hat": Y_hat, "residuals": residuals, "r2": r2}
