import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


data = pd.read_csv('Data/SR_B_trust_trend.csv')


# Preprocessing
X = data.drop('participant_id', axis=1)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

"""

def evaluate_clusters(X, n_clusters, random_state=42):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm_labels = gmm.fit_predict(X)
    

    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)
    kmeans_davies = davies_bouldin_score(X, kmeans_labels)

    gmm_silhouette = silhouette_score(X, gmm_labels)
    gmm_calinski = calinski_harabasz_score(X, gmm_labels)
    gmm_davies = davies_bouldin_score(X, gmm_labels)
    
    return {
        'K-means': {
            'Silhouette Score': kmeans_silhouette,
            'Calinski-Harabasz Index': kmeans_calinski,
            'Davies-Bouldin Index': kmeans_davies
        },
        'GMM': {
            'Silhouette Score': gmm_silhouette,
            'Calinski-Harabasz Index': gmm_calinski,
            'Davies-Bouldin Index': gmm_davies
        }
    }

# Evaluate clusters for different numbers of clusters
best_random_state = None
best_silhouette = -1
n_clusters = 2

for random_state in range(1, 50):
    results = evaluate_clusters(X_scaled, n_clusters, random_state=random_state)
    silhouette = results['K-means']['Silhouette Score']
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_random_state = random_state

print(f"\nNumber of clusters: {n_clusters}")
print(f"Best performing random_state: {best_random_state}")
print(f"Best Silhouette Score: {best_silhouette:.4f}")
print(f"\nNumber of clusters: {n_clusters}")
for method, scores in results.items():
    print(f"\n{method}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# Evaluate stability
def evaluate_stability(X, n_clusters, n_iterations=10):
    kmeans_labels = []
    gmm_labels = []
    
    for _ in range(n_iterations):
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        gmm = GaussianMixture(n_components=n_clusters).fit(X)
        
        kmeans_labels.append(kmeans.labels_)
        gmm_labels.append(gmm.predict(X))
    
    def calculate_stability(labels):
        n_samples = len(labels[0])
        agreement_matrix = np.zeros((n_samples, n_samples))
        
        for l in labels:
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if l[i] == l[j]:
                        agreement_matrix[i, j] += 1
                        agreement_matrix[j, i] += 1
        
        return np.mean(agreement_matrix) / n_iterations
    
    kmeans_stability = calculate_stability(kmeans_labels)
    gmm_stability = calculate_stability(gmm_labels)
    
    return {
        'K-means Stability': kmeans_stability,
        'GMM Stability': gmm_stability
    }

stability_results = evaluate_stability(X_scaled, n_clusters=3)
print("\nStability Results:")
for method, stability in stability_results.items():
    print(f"  {method}: {stability:.4f}")


"""

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plot
def plot_clusters(X, labels, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='Cluster')
    plt.show()

    
kmeans = KMeans(n_clusters=2, random_state=1)
kmeans_labels = kmeans.fit_predict(X_scaled)
gmm = GaussianMixture(n_components=2, random_state=1)
gmm_labels = gmm.fit_predict(X_scaled)


plot_clusters(X_pca, kmeans_labels, 'K-means Clustering')
plot_clusters(X_pca, gmm_labels, 'Gaussian Mixture Model Clustering')


"""

# Elbow
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# BIC scores
bic_scores = []
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=1)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

plt.figure(figsize=(10, 6))
plt.plot(k_range, bic_scores, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC Scores for GMM')
plt.show()

"""