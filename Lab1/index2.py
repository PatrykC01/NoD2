import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Generowanie przykładowych danych
np.random.seed(42) 
X = np.random.rand(1000, 4) * [10, 15, 60, 5]  # [członkowie, doświadczenie, godziny, typ_projektu]
y = np.random.rand(1000, 3) * [100, 50, 180]    # [efektywność, błędy, czas_realizacji]

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
# Zapisujemy historię treningu do zmiennej 'history'
history_multi = model_multi.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
print("--- Zakończono trening modelu ---")

# 5. Przykładowa predykcja
example_input = np.array([[5, 10, 45, 3]])
predicted_output = model_multi.predict(example_input)
print("\n--- Przykładowa predykcja ---")
print(f"Dane wejściowe: {example_input[0]}")
print(f"Przewidywana efektywność: {predicted_output[0][0]:.2f}")
print(f"Przewidywana liczba błędów: {predicted_output[0][1]:.2f}")
print(f"Przewidywany czas realizacji: {predicted_output[0][2]:.2f}")


# 6. Wykres: Historia uczenia (Learning Curve)
plt.figure(figsize=(10, 6))
plt.plot(history_multi.history['loss'], label='Błąd treningowy (loss)')
plt.plot(history_multi.history['val_loss'], label='Błąd walidacyjny (val_loss)')
plt.title('Historia uczenia modelu')
plt.xlabel('Epoka')
plt.ylabel('Wartość błędu (MSE)')
plt.legend()
plt.grid(True)
plt.show()


