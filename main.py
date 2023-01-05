import json
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, rand_score
from sentence_transformers import SentenceTransformer


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = []
        self.clusters = []

    def fit(self, embeddings, max_iter=None):
        # initialize centroids randomly
        self.centroids = embeddings[np.random.choice(embeddings.shape[0], self.n_clusters, replace=False)]

        # initialize cluster assignments
        self.clusters = np.zeros(embeddings.shape[0])

        # run kmeans until convergence
        while True:
            prev_clusters = self.clusters.copy()

            # update each embedding's cluster
            self.clusters = self(embeddings)

            # update centroids based on mean of points in each cluster
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(embeddings[self.clusters == i], axis=0)

            # check for convergence
            if np.array_equal(prev_clusters, self.clusters):
                break

            # enforce max iterations if needed
            if max_iter and max_iter > 0:
                max_iter -= 1

    def __call__(self, embeddings):
        # assign each point to closest centroid
        clusters = np.zeros(embeddings.shape[0])
        for i, point in enumerate(embeddings):
            if not isinstance(point, np.ndarray):
                point = point.toarray()
            distances = np.linalg.norm(point - self.centroids, axis=1)
            clusters[i] = np.argmin(distances)
        return clusters


def kmeans_cluster_and_evaluate(data_file, encoding_type):
    print(f'starting kmeans clustering and evaluation with {data_file} and encoding {encoding_type}')

    # read in data
    with open(data_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        labels, sentences = list(zip(*[row for row in reader]))

    # extract features using TfidfVectorizer or SBERT
    if encoding_type == 'TFIDF':
        embeddings = TfidfVectorizer().fit_transform(sentences)
    else:
        embeddings = SentenceTransformer('all-MiniLM-L12-v2').encode(sentences)

    # initialize KMeans with random centroids
    model = KMeans(n_clusters=len(set(labels)))

    # calc scores
    RI_scores = []
    ARI_scores = []

    for i in range(10):
        # fit model to data
        model.fit(embeddings)

        # evaluate model using RI and ARI
        RI_scores.append(rand_score(labels, model(embeddings)))
        ARI_scores.append(adjusted_rand_score(labels, model(embeddings)))

    # return mean evaluation scores
    evaluation_results = {'mean_RI_score': np.mean(RI_scores), 'mean_ARI_score': np.mean(ARI_scores)}

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'], config["encoding_type"])

    for k, v in results.items():
        print(k, v, sep='\t')
