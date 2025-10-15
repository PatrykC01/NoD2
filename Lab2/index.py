import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

# Krok 1: Generowanie danych 
X_raw = np.random.randint(1, 1000, (500, 1))
y = (1 * (X_raw % 3 == 0) + 2 * (X_raw % 5 == 0)).ravel().astype(int)

# Krok 2: Inżynieria Cech
feature1 = (X_raw % 3 == 0).astype(int)
feature2 = (X_raw % 5 == 0).astype(int)
X_features = np.hstack((feature1, feature2))

# Krok 3: Podział danych i trenowanie modelu
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model (Drzewo Decyzyjne) został pomyślnie wytrenowany.")

# Krok 4: Ocena skuteczności modelu na zbiorze testowym
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu na zbiorze testowym: {accuracy:.4f}\n")


def predict_number_class(trained_model, number):
    feature1 = int(number % 3 == 0)
    feature2 = int(number % 5 == 0)
    
    features_vector = np.array([[feature1, feature2]])
    
    prediction = trained_model.predict(features_vector)
    
    return prediction[0]

print("--- Testowanie predykcji dla nowych, pojedynczych liczb ---")

liczby_do_sprawdzenia = [99, 15, 7, 20, 12, 100]

for liczba in liczby_do_sprawdzenia:
    przewidziana_klasa = predict_number_class(model, liczba)
    print(f"Liczba: {liczba:>4} -> Przewidziana klasa: {przewidziana_klasa}")
