import numpy as np
from sklearn.linear_model import Ridge

from src.models.ridge import RidgeScratch


def test_ridge_matches_sklearn():
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 4
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    true_intercept = 1.23
    y = X.dot(true_coef) + true_intercept + 0.01 * rng.randn(n_samples)

    alpha = 1.0

    # sklearn fit
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X, y)

    # our ridge (weights = [coefs..., intercept])
    model = RidgeScratch(alpha=alpha)
    model.fit(X, y)
    our_coef = model.weights[:-1]
    our_intercept = model.weights[-1]

    assert np.allclose(our_coef, sklearn_model.coef_, atol=1e-8)
    assert np.allclose(our_intercept, sklearn_model.intercept_, atol=1e-8)
