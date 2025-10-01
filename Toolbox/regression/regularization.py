'''
Regularization methods: Ridge Regression, LASSO and eNet
'''

# Required libraries & packages
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=101
)

ridge = Ridge(alpha=1.0).fit(X_train, Y_train)
lasso = Lasso(alpha=0.1).fit(X_train, Y_train)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, Y_train)

#_______________________________________________________________________________________
'''
All of them can be applied directly using the sklearn.linear_model package as above
The foolowing modulation is shown just to keep it consistent with the OLS.py code
Also helps if someone creates a novel method of applying regualrization not yet in python
'''

# Ridge Regression
def ridge_reg(X, Y, alpha=1.0):
    """
    Ridge Regression using sklearn
    alpha = regularization parameter (lambda)
    """
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model

# LASSO
def lasso_reg(X, Y, alpha=0.1):
    """
    LASSO Regression using sklearn.
    alpha = regularization parameter (lambda)
    """
    model = Lasso(alpha=alpha)
    model.fit(X, Y)
    return model

# Elastic Net (balance between Ridge and LASSO)
def eNet(X, Y, alpha=0.1, l1_ratio=0.5):
    """
    Elastic Net Regression using sklearn.
    alpha = regularization parameter
    l1_ratio = balance between LASSO (1) and Ridge (0)
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, Y)
    return model
