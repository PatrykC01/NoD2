import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)  

# 1. Dane: MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

val_split = 0.1
val_size = int(len(x_train_full) * val_split)
x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]

# 2. Inicjalizatory i regularizacja (spójne dla obu modeli)
kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
bias_init = tf.keras.initializers.Zeros()

def create_mlp_model(activation_function='relu'):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation=activation_function,
                     kernel_initializer=kernel_init, bias_initializer=bias_init),
        layers.Dense(10, activation='softmax',
                     kernel_initializer=kernel_init, bias_initializer=bias_init)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Niestandardowy callback do gradientów + scalary
class GradientAndScalarLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, log_dir, tag_prefix="model"):
        super().__init__()
        
        self.x_val = tf.convert_to_tensor(x_val[:64])
        self.y_val = tf.convert_to_tensor(y_val[:64])
        self.writer = tf.summary.create_file_writer(log_dir)
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
       
        with self.writer.as_default():
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    tf.summary.scalar(f"{self.tag_prefix}/{k}", v, step=epoch)
       
        try:
            with tf.GradientTape() as tape:
                preds = self.model(self.x_val, training=False)
                loss = self.model.compiled_loss(self.y_val, preds)
            grads = tape.gradient(loss, self.model.trainable_weights)
            with self.writer.as_default():
                for w, g in zip(self.model.trainable_weights, grads):
                    if g is not None:
                    
                        g_vis = tf.clip_by_value(g, -5.0, 5.0)
                        tf.summary.histogram(f"{self.tag_prefix}/grads/{w.name}", g_vis, step=epoch)
        except Exception as e:
            print(f"[UWAGA] Problem przy logowaniu gradientów w epoce {epoch}: {e}")

# 4. Log directories
base_log_dir = "logs/fit"
os.makedirs(base_log_dir, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_relu = os.path.join(base_log_dir, f"relu_{ts}")
log_dir_sigmoid = os.path.join(base_log_dir, f"sigmoid_{ts}")

tb_cb_relu = tf.keras.callbacks.TensorBoard(log_dir=log_dir_relu, histogram_freq=1, write_graph=True)
tb_cb_sig = tf.keras.callbacks.TensorBoard(log_dir=log_dir_sigmoid, histogram_freq=1, write_graph=True)

grad_cb_relu = GradientAndScalarLoggingCallback(x_val, y_val, log_dir_relu, tag_prefix="relu")
grad_cb_sig = GradientAndScalarLoggingCallback(x_val, y_val, log_dir_sigmoid, tag_prefix="sigmoid")

# 6. Modele
model_relu = create_mlp_model(activation_function='relu')
model_sigmoid = create_mlp_model(activation_function='sigmoid')

print("--- Trenowanie modelu z ReLU ---")
hist_relu = model_relu.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[tb_cb_relu, grad_cb_relu],
    verbose=1
)

print("\n--- Trenowanie modelu z Sigmoid ---")
hist_sig = model_sigmoid.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[tb_cb_sig, grad_cb_sig],
    verbose=1
)

# 7. Ewaluacja na teście
test_loss_relu, test_acc_relu = model_relu.evaluate(x_test, y_test, verbose=0)
test_loss_sig, test_acc_sig = model_sigmoid.evaluate(x_test, y_test, verbose=0)

print(f"\nReLU - Test loss: {test_loss_relu:.4f}, Test acc: {test_acc_relu:.4f}")
print(f"Sigmoid - Test loss: {test_loss_sig:.4f}, Test acc: {test_acc_sig:.4f}")

print("\nUruchom TensorBoard:")
print(f"tensorboard --logdir={base_log_dir}")
