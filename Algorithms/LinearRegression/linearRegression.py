import numpy as np

class LinearRegressor:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, alpha=0.01, epochs=300):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.theta = np.zeros((X.shape[1],1))
        for _ in range(epochs):
            h = X.dot(self.theta)
            error = h - y
            gradient = X.T.dot(error) / len(X)
            self.theta -= alpha * gradient
    
    def predict(self, X: list):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.theta)
    
