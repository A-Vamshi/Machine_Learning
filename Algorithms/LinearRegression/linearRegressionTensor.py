import tensorflow as tf
import numpy as np
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[1], [2], [3], [4], [5]], dtype=float)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_dim=1)])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, y, epochs=1000)
predictions = model.predict(np.array([[6], [7], [8]]))
print(predictions)
