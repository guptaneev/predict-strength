import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="gradient_descent") -> None:
        if method != "normal_equation" and method != "gradient_descent":
            raise AttributeError("invalid parameter passed to method - LinearRegressionScratch constructor")
        self.method = method
    
    def fit(self, X, y):
        ROWS, COLS = X.shape
        X = X.copy()
        if self.method == "normal_equation":
            bias_col = np.ones((ROWS, 1))
            feature_matrix = np.hstack((X, bias_col))
            feature_matrix_t = feature_matrix.T
            
            self.weights = np.linalg.inv((feature_matrix_t @ feature_matrix)) @ (feature_matrix_t @ y)
            
    
    def predict(self, X):
        ROWS, COLS = X.shape
        X = X.copy()
        
        if self.method == "normal_equation":
            bias_col = np.ones((ROWS, 1))
            feature_matrix = np.hstack((X, bias_col))
            return feature_matrix @ self.weights
    