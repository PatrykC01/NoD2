import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# --- Generowanie danych z anomaliami
# 0 = praca normalna, 1 = awaria/anomalia
def generate_lighting_data(n_samples=1000, timesteps=10, anomaly_rate=0.1):
# Tło: normalne fluktuacje napięcia
X = np.random.normal(230, 2, (n_samples, timesteps))
y = np.zeros(n_samples)
n_anomalies = int(anomaly_rate * n_samples)
anomalies = np.random.choice(n_samples, n_anomalies, replace=False)
# Anomalia: nagły spadek napięcia (np. awaria lampy) lub przepięcie
X[anomalies] -= np.random.normal(50, 10, (n_anomalies, timesteps))
y[anomalies] = 1 # Oznaczamy jako awarię
return X.reshape(n_samples, timesteps, 1), y
X, y = generate_lighting_data()
# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- Model LSTM do klasyfikacji binarnej 
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Trening
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# --- Ocena ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Zadanie 13b - Dokładność wykrywania awarii oświetlenia: {accuracy:.2f}")