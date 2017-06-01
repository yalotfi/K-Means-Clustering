import numpy as np  # Maths
from numpy.matlib import repmat  # Repeat boolean vector
from random import sample  # For centroid initialization


def process_data(file_path, K):
    # Import data and create a (m,n) feature matrix, X
    data = np.genfromtxt(fname=file_path, delimiter=",", skip_header=1)
    f1 = np.array(data[:, 1])
    f2 = np.array(data[:, 2])
    feature_matrix = np.vstack((f1, f2)).T
    return feature_matrix


class KMeans(object):
    def __init__(self, X, K):
        super().__init__()
        self.X = X  # Data to be clustered - (m,n) matrix
        self.m = self.X.shape[0]  # Number of examples
        self.n = self.X.shape[1]  # Number of features
        self.K = K  # K clusters to compute
        self.centroids = None  # Store cluster centroids - (K, n) matrix
        self.labels = None  # Store cluster labels - (m, 1) vector
        self.dist_array = None  # Store distances - (1, K) vector

    def _init_centroids(self):
        '''
        Initilize starting centroids at randomly sampled positions
        based on the data.
        '''
        init_index = sample(range(self.m), self.K)
        return np.array([self.X[i] for i in init_index])

    def _squared_dist(self, a, b):
        '''
        Basic Euclidean distance - Squareroot of sum of the squared distances
        '''
        return np.sqrt(np.sum((a - b)**2))

    def _assign_centroids(self):
        '''
        Input a (m,n) matrix, X, and a (K,n) matrix of centroids
        where m is the number of points, K is the number of
        clusters, and n is the number of features.
        '''
        self.centroids = self._init_centroids()
        self.labels = np.zeros((self.m, 1))  # Cluster assignment vector
        # self.dist_array = np.zeros((1, self.K))  # Store distances for each K
        self.dist_array = []
        for i in range(self.m):  # For each i-th x-value in the data
            for j in range(self.K):  # Compute distance for j-th cluster, K
                # self.dist_array[0, j] = squared_dist(self.X[i, :], self.centroids[j, :])
                self.dist_array.append(
                    self._squared_dist(self.X[i, :], self.centroids[j, :])
                )
            min_dist = np.argmin(self.dist_array)  # Save index of min distance
            self.labels[i] = min_dist  # Assign the point, i, to cluster index
        return self.labels  # Return vector of cluster labels

    def _compute_centroids(self, X, labels, K):
        '''
        Input (m,n) matrix, X, and (m,1) label vector to compute new (K,n)
        centroid means.
        '''
        n = X.shape[1]  # Number of features, n, important for maths
        centroids = np.zeros((K, n))  # Return array of dimensions (K,n)
        for k in range(K):
            c = np.equal(labels, k)  # Boolean vector: Label match centroid, k?
            n_k = np.sum(c)  # Integer: Count true values to above question
            C = repmat(c, 1, n)  # Repeat bool vector for linear algebra
            X_c = np.multiply(X, C)  # Element-wise multiplication of X and C
            centroids[k, :] = np.divide(np.sum(X_c, axis=0), n_k)  # Mean Dist
        return centroids

    def run_kmeans(self, max_iter):
        '''
        Starting at a random set of centroids, iterate through the algorithm
        that assigns centroids, calculates the mean distance, then updates
        the cluster assignments. Saves the centroid history for plotting the
        iteration, as well.  Define a max iteration because you will converge
        pretty quickly.
        '''
        centroid_history = []
        current_centroid = self.centroids
        for i in range(max_iter):
            centroid_history.append(current_centroid)
            labels = self._assign_centroids(self.X, current_centroid)
            current_centroid = self._compute_centroids(self.X, labels, self.K)
        return labels, centroid_history


def main(K):
    X = process_data('data/delivery_truck.csv', K)
    kmeans = KMeans(X, K)
    labels = kmeans._assign_centroids()
    print(len(labels))
    # [labels, history] = kmeans.run_kmeans(max_iter=10)
    # print(history[0])
    # print(history[9])


if __name__ == '__main__':
    main(K=3)
