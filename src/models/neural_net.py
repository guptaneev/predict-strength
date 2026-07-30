import numpy as np

class NeuralNetScratch:
    def __init__(self, hidden_units=8, learning_rate = 0.1) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative_fast(sigmoid_output):
        """ Computes derivative using pre-calculated sigmoid outputs.
        Expects: sigmoid_output = 1 / (1 + np.exp(-x)) """
        
        return sigmoid_output * (1 - sigmoid_output)
        
        
    def fit(self, X, y):
        ROWS, COLS = X.shape
        y = y.reshape(-1, 1)
        
        bias_col = np.ones((ROWS, 1))
        feature_matrix = np.hstack((X, bias_col))

        # small random init (not zero) - zero init makes every hidden unit
        # identical forever, since they'd all get the same gradient every step
        rng = np.random.default_rng(seed=42)
        self.W1 = rng.standard_normal((COLS + 1, self.hidden_units)) * 0.01
        self.W2 = rng.standard_normal((self.hidden_units + 1, 1)) * 0.01
        n_iterations = 1000
        self.loss_history = []

        for _ in range(n_iterations):
            z1 = feature_matrix @ self.W1          # (ROWS, hidden_units)
            a1 = self.sigmoid(z1)                  # (ROWS, hidden_units)

            a1_with_bias = np.hstack((a1, bias_col))  # (ROWS, hidden_units + 1)
            z2 = a1_with_bias @ self.W2            # (ROWS, 1)

            dz2 = -2*(y - z2) / ROWS
            dLdW2 = a1_with_bias.T @ dz2
            dLda1 = dz2 @ self.W2[:-1].T
            dz1 = dLda1 * self.sigmoid_derivative_fast(self.sigmoid(z1))
            dLdW1 = feature_matrix.T @ dz1
            
            self.W1 -= self.learning_rate * dLdW1
            self.W2 -= self.learning_rate * dLdW2

            self.loss_history.append(np.mean((y - z2) ** 2))

    def predict(self, X):
        rows = X.shape[0]
        bias_col = np.ones((rows, 1))
        feature_matrix = np.hstack((X, bias_col))

        z1 = feature_matrix @ self.W1
        a1 = self.sigmoid(z1)

        a1_with_bias = np.hstack((a1, np.ones((rows, 1))))
        z2 = a1_with_bias @ self.W2

        return z2.flatten()