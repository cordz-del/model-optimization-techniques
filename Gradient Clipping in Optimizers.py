import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Build a simple model for demonstration
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(1)
])

# Configure the optimizer to clip gradients by norm (e.g., clipnorm=1.0)
optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse')

# Dummy data for training
import numpy as np
X_dummy = np.random.rand(1000, 20)
y_dummy = np.random.rand(1000, 1)

# Train the model with gradient clipping in action
model.fit(X_dummy, y_dummy, epochs=10, batch_size=32)
