import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingScratch:
    def __init__(self, n_estimators, learning_rate, max_depth) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.initial_prediction = y.mean()
        self.trees = []
        self.error_history = []
        current_prediction = np.full(len(y), self.initial_prediction)
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            residual = y - current_prediction
            tree.fit(X, residual)
            current_prediction += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.error_history.append(np.mean(residual**2))
    
    def predict(self, X):
        current_prediction = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            current_prediction += self.learning_rate * tree.predict(X)
        
        return current_prediction
        