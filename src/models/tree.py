import numpy as np

class RegressionTreeNode():
    def __init__(self, feature_idx, threshold, left=None, right=None) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

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

                # skip splits that would leave either child smaller than min_samples_leaf
                if i < self.min_samples_leaf or (n_samples - i) < self.min_samples_leaf:
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
        if depth >= self.max_depth or len(y) <= self.min_samples_leaf or np.var(y) == 0:
            return np.mean(y)
        
        feature_idx, threshold, variance = self._best_split(X, y)
        
        if feature_idx is None:
            return np.mean(y)
        
        mask = X[:, feature_idx] <= threshold
        left_X = X[mask]
        left_y = y[mask]
        right_X = X[~mask]
        right_y = y[~mask]
        
        left = self._build_node(left_X, left_y, depth + 1)
        right = self._build_node(right_X, right_y, depth + 1)
        
        return RegressionTreeNode(feature_idx, threshold, left, right)
    
    def fit(self, X, y):
        self.root = self._build_node(X, y, 0)
    
    def predict(self, X):
        row_preds = []
        for row_index in range(X.shape[0]):
            current = self.root
            row = X[row_index]
            
            while isinstance(current, RegressionTreeNode):
                if row[current.feature_idx] <= current.threshold:
                    current = current.left
                else:
                    current = current.right
                    
            row_preds.append(current)
        
        return np.array(row_preds)
                
        