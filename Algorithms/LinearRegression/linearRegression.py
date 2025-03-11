import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
class LinearRegressor:
    def __init__(self):
        self.theta = 0

    def fit(self, X, y, alpha=0.01, epochs=300):
        self.theta = np.zeros(shape=(X.shape[1],1))
        for _ in range(epochs):
            for j in range(len(self.theta)):
                error = 0
                for i in range(len(X)):
                    h = X.dot(self.theta)
                    error += (h[i][0] - y[i][0]) * (X[i][0])
                self.theta[j] -= alpha * error
    
    def predict(self, X: list):
        ans = []
        for x in X:
            ans.append(self.theta[1][0] * x + self.theta[0][0])
        return ans
    
class LinearRegressorSklearn:
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    
