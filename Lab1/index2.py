import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Generowanie przykładowych danych Z ZALEŻNOŚCIAMI
np.random.seed(42)
# Generowanie danych wejściowych (X)
X_raw = np.random.rand(1000, 4) * [10, 15, 60, 5]  # [członkowie, doświadczenie, godziny, typ_projektu]

# Wyodrębnienie kolumn dla czytelności
czlonkowie = X_raw[:, 0]
doswiadczenie = X_raw[:, 1]
godziny = X_raw[:, 2]
typ_projektu = X_raw[:, 3]

# Tworzenie sztucznych, logicznych zależności dla danych wyjściowych (y)
# Założenie: większe doświadczenie -> większa efektywność, mniej błędów, krótszy czas
# Założenie: trudniejszy projekt -> mniejsza efektywność, więcej błędów, dłuższy czas
efektywnosc = 5 * doswiadczenie + 2 * czlonkowie - 10 * typ_projektu - 0.5 * godziny + np.random.randn(1000) * 5
liczba_bledow = 20 - 2 * doswiadczenie + 3 * typ_projektu + 0.2 * godziny + 0.5 * czlonkowie + np.random.randn(1000) * 3
czas_realizacji = 100 - 4 * doswiadczenie - 8 * czlonkowie + 15 * typ_projektu - 0.8 * godziny + np.random.randn(1000) * 10

y_raw = np.column_stack([efektywnosc, liczba_bledow, czas_realizacji])

# Skalowanie wyników
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_x.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# 2. Budowa modelu sieci neuronowej
model_multi = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3)
])

# 3. Kompilacja modelu
model_multi.compile(optimizer='adam', loss='mean_squared_error')

# 4. Trenowanie modelu
print("\n--- Rozpoczynanie treningu modelu regresji wielowymiarowej ---")

history_multi = model_multi.fit(X_scaled, y_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
print("--- Zakończono trening modelu ---")

# 5. Przykładowa predykcja
example_input_raw = np.array([[5, 10, 45, 3]])

example_input_scaled = scaler_x.transform(example_input_raw)

predicted_output_scaled = model_multi.predict(example_input_scaled)

predicted_output_real = scaler_y.inverse_transform(predicted_output_scaled)

print("\n--- Przykładowa predykcja ---")
print(f"Dane wejściowe (surowe): {example_input_raw[0]}")
print(f"Przewidywana efektywność: {predicted_output_real[0][0]:.2f} (w oryginalnej skali)")
print(f"Przewidywana liczba błędów: {predicted_output_real[0][1]:.2f} (w oryginalnej skali)")
print(f"Przewidywany czas realizacji: {predicted_output_real[0][2]:.2f} (w oryginalnej skali)")


# 6. Wykres: Historia uczenia (Learning Curve)
plt.figure(figsize=(10, 6))
plt.plot(history_multi.history['loss'], label='Błąd treningowy (loss)')
plt.plot(history_multi.history['val_loss'], label='Błąd walidacyjny (val_loss)')
plt.title('Historia uczenia modelu (dane z zależnościami)')
plt.xlabel('Epoka')
plt.ylabel('Wartość błędu (MSE)')
plt.legend()
plt.grid(True)
plt.show()
