import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Seed dla powtarzalności
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Przygotowanie danych
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# 2. Funkcja tworząca model MLP
def create_mlp_model(activation_function='relu'):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation=activation_function),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Konfiguracja logów dla TensorBoard
log_dir_relu = "logs/relu"
log_dir_sigmoid = "logs/sigmoid"

tensorboard_relu = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir_relu, 
    histogram_freq=1
)
tensorboard_sigmoid = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir_sigmoid, 
    histogram_freq=1
)

# 4. Trening modelu z ReLU
print("=== Model z ReLU ===")
model_relu = create_mlp_model(activation_function='relu')
model_relu.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_relu],
    verbose=1
)

# 5. Trening modelu z Sigmoid
print("\n=== Model z Sigmoid ===")
model_sigmoid = create_mlp_model(activation_function='sigmoid')
model_sigmoid.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_sigmoid],
    verbose=1
)

# 6. Ewaluacja
test_loss_relu, test_acc_relu = model_relu.evaluate(x_test, y_test, verbose=0)
test_loss_sig, test_acc_sig = model_sigmoid.evaluate(x_test, y_test, verbose=0)

print(f"\n=== Wyniki na zbiorze testowym ===")
print(f"ReLU    - Loss: {test_loss_relu:.4f}, Accuracy: {test_acc_relu:.4f}")
print(f"Sigmoid - Loss: {test_loss_sig:.4f}, Accuracy: {test_acc_sig:.4f}")

print("\n=== Uruchom TensorBoard: ===")
print("tensorboard --logdir=logs")
