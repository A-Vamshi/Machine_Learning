from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[1], [2], [3], [4], [5]], dtype=float)
model = LinearRegression()
model.fit(X, y)
res = model.predict(X)
print(res)