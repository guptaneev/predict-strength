import numpy as np
from sklearn.linear_model import LinearRegression


def normal_equation(X, y):
    # add intercept column
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    # closed-form solution: (X^T X)^{-1} X^T y
    xtx = Xb.T @ Xb
    xty = Xb.T @ y
    w = np.linalg.inv(xtx) @ xty
    return w


def test_normal_equation_matches_sklearn():
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 4
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    true_intercept = 1.23
    y = X.dot(true_coef) + true_intercept + 0.01 * rng.randn(n_samples)

    # sklearn fit
    lr = LinearRegression()
    lr.fit(X, y)
    sklearn_intercept = lr.intercept_
    sklearn_coef = lr.coef_

    # our normal equation (returns [intercept, coefs...])
    w = normal_equation(X, y)
    our_intercept = w[0]
    our_coef = w[1:]

    # compare intercept and coefficients
    assert np.allclose(our_intercept, sklearn_intercept, atol=1e-8)
    assert np.allclose(our_coef, sklearn_coef, atol=1e-8)
