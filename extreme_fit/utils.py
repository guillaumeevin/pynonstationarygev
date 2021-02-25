import numpy as np
from sklearn.linear_model import LinearRegression


def fit_linear_regression(x, y):
    X = np.array(x).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    a = reg.coef_
    b = reg.intercept_
    return a, b, r2_score
