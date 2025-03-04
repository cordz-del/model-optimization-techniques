import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Build a simple model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define pruning parameters and apply pruning wrapper
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                             final_sparsity=0.5,
                                                             begin_step=0,
                                                             end_step=1000)
}
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the pruned model
model_for_pruning.fit(x_train, y_train, epochs=10, validation_split=0.1)

# Strip pruning wrappers for exporting the final model
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
final_model.save('pruned_model.h5')
