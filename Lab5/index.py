import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Funkcja pomocnicza do drukowania separatora
def print_separator(task_number, task_title):
    print("\n" + "="*80)
    print(f"ZADANIE {task_number}: {task_title}")
    print("="*80 + "\n")

# ==============================================================================
# ZADANIE 1: Warstwa gęsta (Dense) – Trenowanie sieci IRIS bez normalizacji
# ==============================================================================
print_separator(1, "Trenowanie sieci IRIS bez normalizacji danych wejściowych")

iris = load_iris()
X, y = iris.data, iris.target

# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model gęsty
model_iris = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_iris.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

history_iris = model_iris.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
print("Trenowanie modelu IRIS zakończone.")

loss_iris, acc_iris = model_iris.evaluate(X_test, y_test, verbose=0)
print(f"Dokładność na zbiorze testowym (bez normalizacji): {acc_iris:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(history_iris.history['accuracy'], label='Dokładność treningowa')
plt.plot(history_iris.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Zadanie 1: Proces uczenia bez normalizacji danych IRIS')
plt.legend()
plt.show()


# ==============================================================================
# ZADANIE 2: Warstwa konwolucyjna (Conv2D) – Dropout na MNIST
# ==============================================================================
print_separator(2, "Eksperymentowanie z Dropout po warstwie poolingowej na MNIST")

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
x_train_mnist = x_train_mnist[..., tf.newaxis] / 255.0
x_test_mnist = x_test_mnist[..., tf.newaxis] / 255.0

model_mnist = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_mnist.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
history_mnist = model_mnist.fit(x_train_mnist, y_train_mnist, epochs=5, validation_data=(x_test_mnist, y_test_mnist), verbose=0)
print("Trenowanie modelu MNIST zakończone.")

loss_mnist, acc_mnist = model_mnist.evaluate(x_test_mnist, y_test_mnist, verbose=0)
print(f"Dokładność na zbiorze testowym (z Dropout): {acc_mnist:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(history_mnist.history['accuracy'], label='Dokładność treningowa')
plt.plot(history_mnist.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Zadanie 2: Proces uczenia MNIST z warstwami Dropout')
plt.legend()
plt.show()

# ==============================================================================
# ZADANIE 3: Warstwa rekurencyjna (LSTM) – Recurrent Dropout na IMDB
# ==============================================================================
print_separator(3, "Użycie Recurrent Dropout w warstwie LSTM dla IMDB")

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=10000)

x_train_imdb = tf.keras.preprocessing.sequence.pad_sequences(x_train_imdb, maxlen=500)
x_test_imdb = tf.keras.preprocessing.sequence.pad_sequences(x_test_imdb, maxlen=500)

# Budowa sieci rekurencyjnej z Recurrent Dropout
model_imdb = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32),
    tf.keras.layers.LSTM(64, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_imdb.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
history_imdb = model_imdb.fit(x_train_imdb, y_train_imdb, epochs=5, validation_data=(x_test_imdb, y_test_imdb), verbose=1)
print("Trenowanie modelu IMDB zakończone.")

loss_imdb, acc_imdb = model_imdb.evaluate(x_test_imdb, y_test_imdb, verbose=1)
print(f"Dokładność na zbiorze testowym (z Recurrent Dropout): {acc_imdb:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(history_imdb.history['accuracy'], label='Dokładność treningowa')
plt.plot(history_imdb.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Zadanie 3: Proces uczenia IMDB z Recurrent Dropout')
plt.legend()
plt.show()


# ==============================================================================
# ZADANIE 4: Warstwa Transformer – FeedForward bez funkcji aktywacji
# ==============================================================================
print_separator(4, "Implementacja Transformer FeedForward bez funkcji aktywacji")

X_trans = np.random.rand(1000, 10, 512)
Y_trans = np.random.rand(1000, 10, 512)

# Budowa modelu Transformer
input_layer = tf.keras.layers.Input(shape=(10, 512))
attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(input_layer, input_layer)
output = tf.keras.layers.Dense(512, activation=None)(attention_output)

model_transformer = tf.keras.Model(inputs=input_layer, outputs=output)

model_transformer.compile(optimizer='adam', loss='mse')
history_transformer = model_transformer.fit(X_trans, Y_trans, epochs=10, batch_size=32, verbose=0)
print("Trenowanie modelu Transformer zakończone.")

loss_transformer = model_transformer.evaluate(X_trans, Y_trans, verbose=0)
print(f"Błąd średniokwadratowy na danych (z liniową aktywacją): {loss_transformer:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(history_transformer.history['loss'], label='Strata treningowa')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Zadanie 4: Proces uczenia Transformer z liniową warstwą FeedForward')
plt.legend()
plt.show()
