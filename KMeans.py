import numpy as np
import matplotlib.pyplot as plt
from random import sample


def process_data(file_path, K):
    # Import data and create a (m,n) feature matrix, X
    data = np.genfromtxt(fname=file_path, delimiter=",", skip_header=1)
    f1 = np.array(data[:, 1])
    f2 = np.array(data[:, 2])
    feature_matrix = np.vstack((f1, f2)).T
    return feature_matrix


def init_centroids(X, K):
    m = X.shape[0]
    indices = sample(range(0, m), K)
    return np.array([X[i] for i in indices])


def assign_centroids(X, centroids):
    '''
    Input a (m,n) matrix, X, and a (K,n) matrix of centroids
    where m is the number of points, K is the number of
    clusters, and n is the number of features.
    '''
    K = centroids.shape[0]  # Number of clusters/centroids
    m = X.shape[0]  # Number of points/examples
    # n = X.shape[1]  # Number of features
    labels = np.zeros((m, 1))  # Cluster assignment vector
    for i in range(m):  # For each i-th x-value in the data
        dist_array = np.zeros((1, K))  # Stores squared distances for each K
        for j in range(K):  # Compute squared distance for j-th cluster, K
            dist_array[0, j] = np.sqrt(np.sum((X[i, :] - centroids[j, :])**2))
        min_dist = np.argmin(dist_array)  # Save index of min distance
        labels[i] = min_dist  # Assign the point, i, to cluster index
    return labels  # Return vector of cluster labels


def compute_centroids(X, labels, K):
    '''
    Input (m,n) matrix, X, and (m,1) label vector to compute new (K,n)
    centroid means.
    '''
    n = X.shape[1]  # Number of features, n, important for maths
    centroids = np.zeros((K, n))  # Return array of dimensions (K,n)
    for k in range(K):
        c = np.equal(labels, k)  # Boolean vector: Label match centroid, k?
        n_k = np.sum(c)  # Integer: Count true values to above question
        C = np.matlib.repmat(c, 1, n)  # Repeat bool vector for linear algebra
        X_c = np.multiply(X, C)  # Element-wise multiplication of X and C
        centroids[k, :] = np.divide(np.sum(X_c, axis=0), n_k)  # Means Dist
    return centroids


def run_KMeans(X, centroids, max_iter, K):
    '''
    Starting at a random set of centroids, iterate through the algorithm
    that assigns centroids, calculates the mean distance, then updates
    the cluster assignments. Saves the centroid history for plotting the
    iteration, as well.  Define a max iteration because you will converge
    pretty quickly.
    '''
    centroid_history = []
    current_centroid = centroids
    for i in range(max_iter):
        centroid_history.append(current_centroid)
        labels = assign_centroids(X, current_centroid)
        current_centroid = compute_centroids(X, labels, K)
    return labels, centroid_history


def plot_data(X, history, labels):
    '''
    Plot the clustered data and the path of each centroid
    '''
    # Color mapping based on labels
    # Plot data and color by cluster
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()
    # Plot past and final centroids


def main(K):
    X = process_data('data/delivery_truck.csv', K)
    centroids = init_centroids(X, K)
    [labels, history] = run_KMeans(X, centroids, max_iter=10, K=K)
    print(np.unique(labels))
    plot_data(X, history, labels)


if __name__ == '__main__':
    main(K=2)
