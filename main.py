import json
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, rand_score

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X):
        # initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        # initialize cluster assignments
        self.clusters = np.zeros(X.shape[0])

        # run kmeans until convergence
        while True:
            prev_clusters = self.clusters.copy()

            # assign each point to closest centroid
            for i, point in enumerate(X):
                distances = np.linalg.norm(point.toarray() - self.centroids, axis=1)
                self.clusters[i] = np.argmin(distances)

            # update centroids based on mean of points in each cluster
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(X[self.clusters == i], axis=0)

            # check for convergence
            if np.array_equal(prev_clusters, self.clusters):
                break


    def predict(self, X):
        # assign each point to closest centroid
        clusters = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            distances = np.linalg.norm(point.toarray() - self.centroids, axis=1)
            clusters[i] = np.argmin(distances)
        return clusters


def kmeans(data, num_clusters, max_iter=100):
    # Initialize k centroids randomly from the data  points
    centroids = data[np.random.choice(data.shape[0], size=num_clusters, replace=False)]

    for i in range(max_iter):
        # Assign each data point to the closest centroid
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        # Recompute the centroids as the mean of the data points assigned to each centroid
        new_centroids = np.array([data[cluster_labels == k].mean(axis=0) for k in range(num_clusters)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return cluster_labels


def kmeans_cluster_and_evaluate(data_file, encoding_type):
    # todo: implement this function
    print(f'starting kmeans clustering and evaluation with {data_file} and encoding {encoding_type}')
    data = {}
    # read in data
    with open(data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        data['sentences'] = [row[1] for row in reader]

    with open(data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        data['labels'] = [row[0] for row in reader]

    # extract features using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['sentences'])

    # initialize KMeans with random centroids
    model = KMeans(n_clusters=len(set(data['labels'])))

    # fit model to data
    model.fit(X)

    # evaluate model using RI and ARI
    RI_scores = []
    ARI_scores = []
    for i in range(10):
        RI_scores.append(rand_score(data['labels'], model.predict(X)))
        ARI_scores.append(adjusted_rand_score(data['labels'], model.predict(X)))

    # todo: perform feature extraction from sentences and
    #  write your own kmeans implementation with random centroids initialization

    # todo: evaluate against known ground-truth with RI and ARI:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html and
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html


    # todo: fill in the dictionary below with evaluation scores averaged over 10 invocations

    # return mean evaluation scores
    evaluation_results = {'mean_RI_score': np.mean(RI_scores), 'mean_ARI_score': np.mean(ARI_scores)}

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'], config["encoding_type"])

    for k, v in results.items():
        print(k, v)
