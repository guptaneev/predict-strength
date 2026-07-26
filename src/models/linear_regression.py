import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="gradient_descent", num_iterations = 1000, learning_rate = 1e-6) -> None:
        if method != "normal_equation" and method != "gradient_descent":
            raise AttributeError("invalid parameter passed to method - LinearRegressionScratch constructor")
        self.method = method
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        ROWS, COLS = X.shape
        X = X.copy()
        
        bias_col = np.ones((ROWS, 1))
        feature_matrix = np.hstack((X, bias_col))
        feature_matrix_t = feature_matrix.T
                    
        if self.method == "normal_equation":
            self.weights = np.linalg.inv((feature_matrix_t @ feature_matrix)) @ (feature_matrix_t @ y)
            
        elif self.method == "gradient_descent":
            self.weights = np.zeros(COLS + 1) # COLS features + BIAS
            self.loss_history = []
            
            for _ in range(self.num_iterations):
                predicted = feature_matrix @ self.weights
                loss = y - predicted
                self.loss_history.append(np.mean(loss**2))
                gradient = (-2 / ROWS) * (feature_matrix_t @ loss)
                self.weights -= self.learning_rate * gradient
                
        else:
            raise AttributeError("invalid parameter passed to method - LinearRegressionScratch constructor")
            
    
    def predict(self, X):
        ROWS, COLS = X.shape
        X = X.copy()
        
        bias_col = np.ones((ROWS, 1))
        feature_matrix = np.hstack((X, bias_col))
        return feature_matrix @ self.weights