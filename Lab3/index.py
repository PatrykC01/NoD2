# ==============================================================================
# SEKCJA 0: IMPORT BIBLIOTEK I PRZYGOTOWANIE DANYCH
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.datasets import load_wine

# 1. Wczytanie i przygotowanie własnego zbioru danych
print("\nWczytywanie i przygotowywanie danych YouTube...")
data = pd.read_csv('cleaned_youtube_dataset.csv', sep=",")
data = data.sample(frac=0.1, random_state=42) 

X = data.drop('liked', axis=1)
y = data['liked']

categorical_features = ['device', 'watch_time_of_day']
X = pd.get_dummies(X, columns=categorical_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ==============================================================================
# ZADANIE 1: PORÓWNANIE DOKŁADNOŚCI MODELI BAZOWYCH
# ==============================================================================
print("\nZadanie 1: Porównywanie modeli na domyślnych ustawieniach...")

# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Dokładność RandomForest: {acc_rf:.4f}")

# Model 2: XGBoost
xgb_default = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_default.fit(X_train, y_train)
y_pred_xgb_default = xgb_default.predict(X_test)
acc_xgb_default = accuracy_score(y_test, y_pred_xgb_default)
print(f"Dokładność XGBoost (domyślny): {acc_xgb_default:.4f}")

# Model 3: Stacking
estimators_base = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
]
stack_base = StackingClassifier(estimators=estimators_base, final_estimator=LogisticRegression(), n_jobs=-1)
stack_base.fit(X_train, y_train)
y_pred_stack_base = stack_base.predict(X_test)
acc_stack_base = accuracy_score(y_test, y_pred_stack_base)
print(f"Dokładność Stacking (bazowy): {acc_stack_base:.4f}")

# ==============================================================================
# ZADANIE 2: TUNING HIPERPARAMETRÓW DLA XGBOOST
# ==============================================================================
print("\nZadanie 2: Zaawansowany tuning hiperparametrów dla XGBoost za pomocą RandomizedSearchCV...")

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train, y_train)

print("\nNajlepsze parametry znalezione przez RandomizedSearchCV:")
print(random_search.best_params_)

best_xgb = random_search.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test)
acc_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)

print(f"\nDokładność XGBoost (przed tuningiem): {acc_xgb_default:.4f}")
print(f"Dokładność XGBoost (PO ZAAWANSOWANYM tuningu): {acc_xgb_tuned:.4f}")

# ==============================================================================
# ZADANIE 3: WPROWADZENIE NOWEGO MODELU DO ZESTAWU STACKINGOWEGO
# ==============================================================================
print("\nZadanie 3: Tworzenie nowego modelu Stacking z dostrojonym XGBoost i knn...")

estimators_new = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb_tuned', best_xgb),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)))
]
stack_new = StackingClassifier(estimators=estimators_new, final_estimator=LogisticRegression(), n_jobs=-1)
stack_new.fit(X_train, y_train)
y_pred_stack_new = stack_new.predict(X_test)
acc_stack_new = accuracy_score(y_test, y_pred_stack_new)
print(f"Dokładność nowego StackingClassifier: {acc_stack_new:.4f}")

# ==============================================================================
# ZADANIE 5: FINALNY WYKRES PORÓWNAWCZY DLA DANYCH YOUTUBE
# ==============================================================================
print("\nZadanie 5: Generowanie finalnego wykresu porównawczego...")

models_final = ['Random Forest', 'XGBoost (Default)', 'Stacking (Base)', 'XGBoost (Tuned)', 'Stacking (New)']
accuracies_final = [acc_rf, acc_xgb_default, acc_stack_base, acc_xgb_tuned, acc_stack_new]

plt.figure(figsize=(12, 7))
bars = plt.bar(models_final, accuracies_final, color=['skyblue', 'salmon', 'lightgreen', 'red', 'mediumpurple'])
plt.ylabel('Dokładność')
plt.title('Finalne porównanie dokładności modeli na danych YouTube')

plt.ylim(min(accuracies_final) - 0.005, max(accuracies_final) + 0.005)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nAnaliza ważności cech na podstawie najlepszego modelu XGBoost...")


importances = best_xgb.feature_importances_

feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

top_n = 10
top_features = feature_importance_df.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(top_features['feature'], top_features['importance'], color='c')
plt.xlabel('Ważność (Importance)')
plt.ylabel('Cecha (Feature)')
plt.title(f'Top {top_n} najważniejszych cech według modelu XGBoost')

plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# ==============================================================================
# ZADANIE 4: PRZETESTOWANIE MODELI NA INNYM ZBIORZE DANYCH (WINE)
# ==============================================================================
print("\n\nZadanie 4: Rozpoczynanie testów na zbiorze danych 'Wine'...")

# 1. Wczytanie i przygotowanie danych Wine
wine_data = load_wine()
X_w, y_w = wine_data.data, wine_data.target

scaler = StandardScaler()
X_w_scaled = scaler.fit_transform(X_w)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w_scaled, y_w, test_size=0.2, random_state=42)

# 2. Trenowanie modeli
rf_wine = RandomForestClassifier(random_state=42).fit(X_train_w, y_train_w)
acc_rf_w = accuracy_score(y_test_w, rf_wine.predict(X_test_w))
print(f"Dokładność RandomForest na zbiorze Wine: {acc_rf_w:.4f}")

xgb_wine = XGBClassifier(eval_metric='mlogloss', random_state=42).fit(X_train_w, y_train_w)
acc_xgb_w = accuracy_score(y_test_w, xgb_wine.predict(X_test_w))
print(f"Dokładność XGBoost na zbiorze Wine: {acc_xgb_w:.4f}")

estimators_wine = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]
stack_wine = StackingClassifier(estimators=estimators_wine, final_estimator=LogisticRegression()).fit(X_train_w, y_train_w)
acc_stack_w = accuracy_score(y_test_w, stack_wine.predict(X_test_w))
print(f"Dokładność Stacking na zbiorze Wine: {acc_stack_w:.4f}")

# 3. Wykres dla zbioru Wine
models_wine = ['Random Forest', 'XGBoost', 'Stacking']
accuracies_wine = [acc_rf_w, acc_xgb_w, acc_stack_w]

plt.figure(figsize=(10, 6))
plt.bar(models_wine, accuracies_wine, color=['blue', 'green', 'orange'])
plt.ylabel('Dokładność')
plt.title('Porównanie dokładności modeli na zbiorze danych Wine')
plt.ylim(0.9, 1.01)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
