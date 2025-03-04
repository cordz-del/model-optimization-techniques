import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

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

# Define a learning rate scheduler function
def scheduler(epoch, lr):
    # Decay the learning rate by 10% every 10 epochs
    if epoch != 0 and epoch % 10 == 0:
        return lr * 0.9
    return lr

# Create callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(scheduler)

# Train the model using the callbacks
model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop, lr_schedule])
