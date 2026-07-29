import numpy as np

class RegressionTreeScratch():
    def __init__(self, max_depth, min_samples_leaf) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_variance = float("inf")

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            sort_order = np.argsort(feature_values)
            sorted_values = feature_values[sort_order]
            sorted_y = y[sort_order]

            for i in range(1, n_samples):
                # candidate thresholds only make sense between two different values -
                # splitting between two identical values doesn't separate anything
                if sorted_values[i] == sorted_values[i - 1]:
                    continue

                left_y = sorted_y[:i]
                right_y = sorted_y[i:]

                weighted_variance = (
                    (len(left_y) / n_samples) * np.var(left_y)
                    + (len(right_y) / n_samples) * np.var(right_y)
                )

                if weighted_variance < best_variance:
                    best_variance = weighted_variance
                    best_feature = feature_idx
                    # threshold = midpoint between the two straddling values
                    best_threshold = (sorted_values[i - 1] + sorted_values[i]) / 2

        return best_feature, best_threshold, best_variance
                
            
    def _build_node(self, X, y, depth):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
        