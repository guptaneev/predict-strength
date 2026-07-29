import numpy as np

from src.models.boosting import GradientBoostingScratch


def test_training_error_strictly_decreases():
    rng = np.random.RandomState(0)

    n_samples = 200
    X = rng.uniform(0, 10, size=(n_samples, 2))
    y = np.where(X[:, 0] < 5, 10, 50) + np.where(X[:, 1] < 5, 0, 20) + rng.normal(0, 0.5, size=n_samples)

    model = GradientBoostingScratch(n_estimators=20, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    errors = model.error_history
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
