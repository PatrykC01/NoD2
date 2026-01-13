import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set(style="whitegrid")

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
df = pd.read_csv(url)
df.columns = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAgg', 'FineAgg', 'Age', 'Strength']
features = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAgg', 'FineAgg', 'Age']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# --- WYKRES 1: Metoda Łokcia i Silhouette ---

inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(k_range, inertia, 'bo-', label='Inertia')
ax1.set_xlabel('Liczba skupień (k)')
ax1.set_ylabel('Inertia', color='b')
ax1.set_title('Wykres 1: Metoda Łokcia i Wskaźnik Silhouette')
ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, 'ro--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
plt.show() 

# --- WYKRES 2: Dendrogram ---


plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, no_labels=True)
plt.title('Wykres 2: Dendrogram (Metoda Warda)')
plt.xlabel('Próbki')
plt.ylabel('Odległość')
plt.show() 

# --- ALGORYTMY I PORÓWNANIE ---

best_k = 8

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

print("\n--- Testowanie parametrów DBSCAN ---")
eps_values = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
best_eps = 0.5
best_score = -1

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=16)
    labels = db.fit_predict(X_scaled)
    
    unique_labels = set(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(X_scaled):
        score = silhouette_score(X_scaled, labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"eps={eps}: Klastry={n_clusters}, Szum={n_noise}, Silhouette={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_eps = eps
    else:
        print(f"eps={eps}: Brak sensownego podziału (tylko szum lub 1 grupa)")

db_final = DBSCAN(eps=best_eps, min_samples=16)
labels_dbscan = db_final.fit_predict(X_scaled)

agg_clustering = AgglomerativeClustering(n_clusters=best_k, metric='euclidean', linkage='ward')
labels_agg = agg_clustering.fit_predict(X_scaled)

results = []

def evaluate_model(name, labels, X):
    if len(set(labels)) < 2: return [name, np.nan, np.nan, np.nan]
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return [name, sil, ch, db]

results.append(evaluate_model("K-means", labels_kmeans, X_scaled))
results.append(evaluate_model("DBSCAN", labels_dbscan, X_scaled))
results.append(evaluate_model("Agglomerative", labels_agg, X_scaled))

print("\n--- Porównanie Metod ---")
print(pd.DataFrame(results, columns=["Algorytm", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]))

# --- WYKRES 3: Scatter Plot K-means ---

plt.figure(figsize=(10, 6))
plt.scatter(df['Cement'], df['Water'], c=labels_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.title(f'Wykres 3: Wynik K-means (k={best_k}): Cement vs Woda')
plt.xlabel('Cement [kg]')
plt.ylabel('Woda [kg]')
plt.colorbar(label='Numer klastra')
plt.show() 

# --- WYKRES 4: Heatmapa Profilu ---

df['Cluster'] = labels_kmeans
cluster_mean = df.groupby('Cluster').mean()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_mean, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=.5)
plt.title(f'Wykres 4: Profil Klastrów (Średnie wartości składników i wytrzymałości)')
plt.show()
