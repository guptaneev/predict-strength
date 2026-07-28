import numpy as np

class RidgeScratch:
    def __init__(self, alpha) -> None:
        self.alpha = alpha
    
    def fit(self, X, y):
            ROWS, COLS = X.shape
            X = X.copy()
            
            bias_col = np.ones((ROWS, 1))
            feature_matrix = np.hstack((X, bias_col))
            feature_matrix_t = feature_matrix.T
            
            identity = np.identity(COLS + 1)
            # we don't want to penalize the bias term for being large
            identity[-1][-1] = 0
                        
            self.weights = np.linalg.inv(( (feature_matrix_t @ feature_matrix) + (self.alpha * identity) )) @ (feature_matrix_t @ y)
                

    def predict(self, X):
        X = X.copy()
                
        bias_col = np.ones((X.shape[0], 1))
        feature_matrix = np.hstack((X, bias_col))
        return feature_matrix @ self.weights
    