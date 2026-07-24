import numpy as np

# baseline mean absolute error "prediction"
def mae(actual, predicted) -> float:
    errors = actual - predicted
    abs_errors = np.abs(errors)
    return np.mean(abs_errors)