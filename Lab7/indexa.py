import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- Generowanie danych (symulacja hałasu w dB) ---

def generate_noise_data(n_samples=1000, freq=0.05, noise=0.2):
    x = np.arange(n_samples)
    # Symulacja: baza (np. 60dB) + zmienność sinusoidalna + szum losowy
    y = 60 + 10 * np.sin(2*np.pi*freq*x) + np.random.normal(0, noise*10, size=n_samples)
    return y.reshape(-1, 1)

data = generate_noise_data()

# --- Normalizacja ---
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# --- Przygotowanie danych sekwencyjnych (Listing 1) [cite: 13] ---
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: (i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 20
X, y = create_dataset(data_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Model LSTM ---
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Trening
model.fit(X, y, epochs=15, batch_size=16, verbose=1)

# --- Prognozowanie i Wykres ---
predicted = model.predict(X)
predicted_real = scaler.inverse_transform(predicted)
y_real = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_real[:200], label='Rzeczywisty poziom hałasu (dB)')
plt.plot(predicted_real[:200], label='Prognozowany poziom hałasu (dB)', linestyle='--')
plt.title('Zadanie 13a: Prognozowanie hałasu ulicznego')
plt.ylabel('Decybele (dB)')
plt.xlabel('Czas')
plt.legend()
plt.show()