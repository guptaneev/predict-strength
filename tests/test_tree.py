import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.models.tree import RegressionTreeScratch


def test_tree_matches_sklearn():
    rng = np.random.RandomState(0)

    # small synthetic dataset - piecewise-constant-ish so a shallow tree can
    # actually capture the pattern, easier to sanity check against sklearn
    n_samples = 200
    X = rng.uniform(0, 10, size=(n_samples, 2))
    y = np.where(X[:, 0] < 5, 10, 50) + np.where(X[:, 1] < 5, 0, 20) + rng.normal(0, 0.5, size=n_samples)

    max_depth = 3
    min_samples_leaf = 5

    # sklearn fit
    sklearn_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)

    # our tree
    model = RegressionTreeScratch(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.allclose(predictions, sklearn_predictions, atol=1e-6)
