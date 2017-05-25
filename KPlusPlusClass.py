import numpy as np
import KMeansClass as model


def KMeansPlusPlus(X, K):
    '''
    Use the K-Means++ Algorithm to set initial centroids intelligently.
        1. Choose inital center, c1, uniformly at random from X
        2. Compute distance vector, D, between each point in X
        3. Choose new center from weighted probability distirbution
        4. Repeat 2 and 3 until k-centers chosen
        5. Run KMeans Clustering
    '''
    m = X.shape[0]
    c1 = sample(range(m), 1)
    D = np.zeros((m, 1))
    for i in range(m):
        D[i] = squared_dist(X[i, :], c1)
    return D


def main(K):
    X = process_data('data/delivery_truck.csv', K)
    kmeans = model.KMeans(X, K)
    [labels, history] = kmeans.run_kmeans(max_iter=10)
    print(history[0])
    print(history[9])


if __name__ == '__main__':
    main(K=4)


