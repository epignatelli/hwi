from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from sklearn import metrics

def cluster_images(data_dir="../data/train/", n_clusters=3):



    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_clusters).fit(data)
    km = KMeans(init=pca.components_, n_clusters=n_clusters)
    km.fit_transform(data)

    # Getting most representative data points for each cluster
    best_k_index = metrics.pairwise_distances_argmin(pca.transform(km.cluster_centers_), pca.transform(data))
    best_k = df.iloc[best_k_index]
    
    # setting variables
    labels = km.labels_
    n_clusters = n_clusters
    cluster_centers = km.cluster_centers_
    representants = best_k