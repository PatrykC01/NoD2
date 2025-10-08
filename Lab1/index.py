import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Definicja i generowanie danych
def target_function(x):
    return np.sin(x**2)

x_train = np.linspace(1, 3, 500).reshape(-1, 1)
y_train = target_function(x_train)

x_test = np.linspace(1, 3, 100).reshape(-1, 1)
y_test = target_function(x_test)

# 2. Budowa modelu sieci neuronowej
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 3. Kompilacja modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Trenowanie modelu
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), verbose=0)

# 5. Ewaluacja modelu i sprawdzenie warunku błędu
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Finalny błąd (Mean Squared Error) na zbiorze testowym: {loss:.4f}")

if loss < 0.01:
    print("Warunek 'Error < 0.01' został spełniony.")
else:
    print("Warunek 'Error < 0.01' NIE został spełniony.")

# 6. Predykcja i wizualizacja wyników
y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Wartości Oczekiwane vs. Wartości Prognozowane')
plt.xlabel('Wartości Oczekiwane (y_test)')
plt.ylabel('Wartości Prognozowane (y_pred)')
plt.grid(True)
plt.show()