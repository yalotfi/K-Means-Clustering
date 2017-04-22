import numpy as np
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
    index = np.zeros((m, 1))  # Cluster assignment vector
    for i in range(m):  # For each i-th x-value in the data
        dist_array = np.zeros((1, K))  # Stores squared distances for each K
        for j in range(K):  # Compute squared distance for j-th cluster, K
            dist_array[0][j] = np.sum(np.square((X[i]) - centroids[j]))
        min_dist = np.argmin(dist_array[0])  # Save index of min distance
        index[i] = min_dist  # Assign the point, i, to cluster
    return index  # Return vector of cluster indices


def compute_centroids(X, index, K):
    m = X.shape[0]
    n = X.shape[1]
    return [m, n]


def main():
    X = process_data('data/delivery_truck.csv', K=2)
    centroids = init_centroids(X, K=2)
    print('K: {0} and m: {1}'.format(centroids.shape[0], X.shape[0]))
    labels = assign_centroids(X, centroids)
    print(labels)


if __name__ == '__main__':
    main()
