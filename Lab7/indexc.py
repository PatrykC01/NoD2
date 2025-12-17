import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
# --- Generowanie danych wzorców
# Klasy: 0 = Noc/Pustki, 1 = Ruch standardowy, 2 = Szczyt/Tłum
def generate_pedestrian_patterns(n_classes=3, samples_per_class=500, timesteps=20):
X, y = [], []
for cls in range(n_classes):
for _ in range(samples_per_class):
if cls == 0: # Noc (mała wariancja, niskie wartości)
seq = np.random.normal(5, 2, timesteps)
elif cls == 1: # Dzień (sinusoida - regularny ruch)
seq = 20 + 10 * np.sin(np.linspace(0, 3*np.pi, timesteps)) + np.random.normal(0, 2, timesteps)
else: # Szczyt (wysokie wartości, dynamiczne zmiany)
seq = 50 + 15 * np.sin(np.linspace(0, 6*np.pi, timesteps)) + np.random.normal(0, 5, timesteps)
X.append(seq)
y.append(cls)
X = np.array(X).reshape(-1, timesteps, 1)
y = tf.keras.utils.to_categorical(y, num_classes=n_classes)
return X, y
X, y = generate_pedestrian_patterns()
# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- Model LSTM do klasyfikacji wieloklasowej
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Trening
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
# --- Ocena ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Zadanie 13c - Dokładność rozpoznawania wzorców ruchu pieszych: {accuracy:.2f}")