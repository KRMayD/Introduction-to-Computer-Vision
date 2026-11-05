"""Small independent checks before the expensive full-data experiment."""

import unittest

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

from knn_cifar10.core import (ExactKNN, METRICS, evaluate, majority_predictions,
                        distance_weighted_predictions, nearest_tie_predictions)


class KNNTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.x = rng.integers(1, 256, size=(43, 17), dtype=np.uint8)
        self.y = rng.integers(0, 10, size=43)
        self.query = rng.integers(1, 256, size=(13, 17), dtype=np.uint8)
        self.ks = (1, 3, 5, 9)

    def check_backend(self, device):
        classifier = ExactKNN(self.ks, device=device, batch_size=4).fit(self.x, self.y)
        labels = classifier.neighbor_labels(self.query)
        predictions = classifier.predict_all(self.query)
        weighted_labels, neighbor_distances = classifier.neighbor_labels(self.query, return_distances=True)
        for metric in METRICS:
            reference = pairwise_distances(self.query.astype(float), self.x.astype(float), metric=metric)
            order = np.argsort(reference, axis=1, kind="stable")[:, :max(self.ks)]
            np.testing.assert_array_equal(labels[metric], self.y[order])
            np.testing.assert_array_equal(weighted_labels[metric], labels[metric])
            np.testing.assert_allclose(neighbor_distances[metric], np.take_along_axis(reference, order, axis=1), rtol=1e-10, atol=1e-10)
            weighted = distance_weighted_predictions(weighted_labels[metric], neighbor_distances[metric], self.ks)
            for k in self.ks:
                model = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute", n_jobs=1).fit(self.x.astype(float), self.y)
                np.testing.assert_array_equal(predictions[metric][k], model.predict(self.query.astype(float)))
                weighted_model = KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric,
                                                      algorithm="brute", n_jobs=1).fit(self.x.astype(float), self.y)
                np.testing.assert_array_equal(weighted[k], weighted_model.predict(self.query.astype(float)))

    def test_cpu_matches_independent_distance_and_classifier(self):
        self.check_backend("cpu")

    def test_gpu_matches_independent_distance_and_classifier(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("No visible CUDA GPU")
        torch.set_num_threads(4)
        self.check_backend("cuda:0")

    def test_vote_tie_uses_smallest_class(self):
        labels = np.array([[7, 2, 7, 2], [4, 4, 3, 3]])
        predicted = majority_predictions(labels, [1, 2, 3, 4])
        np.testing.assert_array_equal(predicted[2], [2, 4])
        np.testing.assert_array_equal(predicted[4], [2, 3])

    def test_distance_ties_and_zero_cosine(self):
        x = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
        y = np.array([8, 3, 2, 9])
        query = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                devices.append("cuda:0")
        except ImportError:
            pass
        for device in devices:
            labels = ExactKNN([1, 4], device=device).fit(x, y).neighbor_labels(query)
            for metric in METRICS:
                self.assertEqual(labels[metric][0, 0], 8)
            np.testing.assert_array_equal(labels["cosine"][1], y)
            self.assertEqual(labels["euclidean"][1, 0], 9)
            self.assertEqual(labels["manhattan"][1, 0], 9)

    def test_metric_formula(self):
        result = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision_macro"], (1 + 2 / 3) / 10)
        self.assertAlmostEqual(result["recall_macro"], (0.5 + 1) / 10)
        self.assertAlmostEqual(result["f1_macro"], (2 / 3 + 0.8) / 10)

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            ExactKNN([0])
        with self.assertRaises(ValueError):
            ExactKNN([44]).fit(self.x, self.y)

    def test_weighted_zero_distances_override_positive_distances(self):
        labels = np.array([[2, 2, 5, 5, 5]])
        distances = np.array([[0., 0., 0., 1., 2.]])
        result = distance_weighted_predictions(labels, distances, [1, 3, 5])
        for pred in result.values():
            np.testing.assert_array_equal(pred, [2])

    def test_weighted_tie_uses_nearest_winning_class(self):
        pred = distance_weighted_predictions(np.array([[7, 2]]), np.array([[1., 1.]]), [2])
        np.testing.assert_array_equal(pred[2], [7])
        pred = nearest_tie_predictions(np.array([[3, 2, 7, 2, 7]]), [5])
        np.testing.assert_array_equal(pred[5], [2])

    def test_euclidean_weights_use_distance_instead_of_squared_distance(self):
        x = np.array([[1., 0.], [1.5, 0.], [0., 1.5]])
        model = ExactKNN([3], metrics=("euclidean",)).fit(x, [0, 1, 1])
        labels, distances = model.neighbor_labels(np.array([[0., 0.]]), return_distances=True)
        np.testing.assert_allclose(distances["euclidean"], [[1., 1.5, 1.5]])
        pred = distance_weighted_predictions(labels["euclidean"], distances["euclidean"], [3])
        np.testing.assert_array_equal(pred[3], [1])

    def test_weighted_rejects_invalid_distances(self):
        for distances in [np.array([[1., -1.]]), np.array([[1., np.nan]]), np.array([[2., 1.]])]:
            with self.assertRaises(ValueError):
                distance_weighted_predictions(np.array([[0, 1]]), distances, [2])

    def test_nearest_tie_voting_through_k101(self):
        from collections import Counter
        rng = np.random.default_rng(101)
        x = rng.integers(1, 256, size=(127, 19), dtype=np.uint8)
        y = rng.integers(0, 10, size=127)
        query = rng.integers(1, 256, size=(7, 19), dtype=np.uint8)
        ks = list(range(1, 102, 2))
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                devices.append("cuda:0")
        except ImportError:
            pass
        for device in devices:
            neighbors = ExactKNN(ks, device=device, batch_size=3).fit(x, y).neighbor_labels(query)
            for metric in METRICS:
                distances = pairwise_distances(query.astype(float), x.astype(float), metric=metric)
                ordered = y[np.argsort(distances, axis=1, kind="stable")[:, :101]]
                np.testing.assert_array_equal(neighbors[metric], ordered)
                predictions = nearest_tie_predictions(neighbors[metric], ks)
                for k in ks:
                    expected = []
                    for row in ordered[:, :k]:
                        counts = Counter(row)
                        top = max(counts.values())
                        expected.append(next(label for label in row if counts[label] == top))
                    np.testing.assert_array_equal(predictions[k], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
