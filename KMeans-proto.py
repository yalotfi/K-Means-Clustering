import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    # Import data and create feature matrix
    file_path = 'data/delivery_truck.csv'
    data = np.genfromtxt(fname=file_path, delimiter=",", skip_header=1)
    f1 = np.array(data[:, 1])
    f2 = np.array(data[:, 2])
    feature_matrix = np.vstack((f1, f2)).T

    # Fit K-Means
    kmeans = KMeans(n_clusters=4).fit(feature_matrix)

    # Plot Clusters
    x = feature_matrix[:, 0]
    y = feature_matrix[:, 1]
    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()


if __name__ == '__main__':
    main()
