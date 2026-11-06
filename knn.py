"""CIFAR-10 로드, 정확한 최근접 이웃 탐색, 투표와 평가."""

import csv
import pickle
import tarfile
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

METRICS = ("euclidean", "manhattan", "cosine")
VOTING = ("uniform_class_id", "uniform_nearest", "distance_weighted")
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
DEFAULT_K = (1, 3, 5, 7, 9, 11, 15, 21)


def load_cifar10(archive):
    """신뢰할 수 있는 CIFAR-10 Python tar 파일을 메모리에서 읽는다."""
    # 디스크에 압축을 풀지 않는다. 픽셀은 정규화 없이 원래의 uint8 값을 쓴다.
    with tarfile.open(archive, "r:*") as handle:
        def batch(name):
            with handle.extractfile(f"cifar-10-batches-py/{name}") as stream:
                data = pickle.load(stream, encoding="bytes")
            x = np.asarray(data[b"data"])
            y = np.asarray(data[b"labels"], dtype=np.int64)
            if x.shape != (10000, 3072) or x.dtype != np.uint8 or y.shape != (10000,):
                raise ValueError(f"Unexpected CIFAR-10 batch: {name}")
            return x, y

        batches = [batch(f"data_batch_{i}") for i in range(1, 6)]
        x, y = (np.concatenate(parts) for parts in zip(*batches))
        test_x, test_y = batch("test_batch")
    # 학습 50,000장과 공식 테스트 10,000장을 모두 사용했는지 확인한다.
    if not np.array_equal(np.bincount(y, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected 5,000 training images per class.")
    if not np.array_equal(np.bincount(test_y, minlength=10), np.full(10, 1000)):
        raise ValueError("Expected 1,000 test images per class.")
    return x, y, test_x, test_y


class ExactKNN:
    """모든 학습 이미지와 거리를 비교하는 KNN. 근사 탐색은 사용하지 않는다."""

    def __init__(self, k_values=DEFAULT_K, metrics=METRICS, device="cpu", batch_size=128):
        self.ks = tuple(sorted(set(k_values)))
        self.metrics = tuple(metrics)
        self.device, self.batch_size = device, batch_size
        if not self.ks or min(self.ks) < 1 or batch_size < 1:
            raise ValueError("k and batch_size must be positive.")
        if not self.metrics or not set(self.metrics).issubset(METRICS):
            raise ValueError(f"Supported metrics: {METRICS}")
        if device != "cpu" and not device.startswith("cuda:"):
            raise ValueError("Use cpu or a visible CUDA device, e.g. cuda:0.")

    def fit(self, x, y):
        # KNN의 학습은 이미지와 정답을 저장하는 과정이다. 가중치 최적화는 없다.
        x = np.asarray(x)
        self.y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or self.y.shape != (len(x),) or max(self.ks) > len(x):
            raise ValueError("Invalid training shape or k exceeds training size.")
        if not np.isfinite(x).all() or np.any((self.y < 0) | (self.y > 9)):
            raise ValueError("Expected finite features and class IDs 0..9.")
        self.n_features = x.shape[1]
        self.needs_dot = bool({"euclidean", "cosine"}.intersection(self.metrics))
        # CIFAR-10의 L1 합은 3072*255 < 2**24라 float32에서도 정확한 정수다.
        self.l1_float32 = x.dtype == np.uint8 and x.shape[1] * 255 < 2**24
        if self.device != "cpu":
            import torch
            self.torch = torch
            self.x64 = torch.as_tensor(x, dtype=torch.float64, device=self.device) if self.needs_dot else None
            self.norm2 = (self.x64 * self.x64).sum(1) if self.needs_dot else None
            dtype = torch.float32 if self.l1_float32 else torch.float64
            self.x_l1 = torch.as_tensor(x, dtype=dtype, device=self.device) if "manhattan" in self.metrics else None
        else:
            self.x64 = np.asarray(x, dtype=np.float64)
            self.norm2 = np.einsum("ij,ij->i", self.x64, self.x64)
        return self

    def neighbor_labels(self, query, return_distances=False):
        """거리순 이웃 클래스 배열 (질의 수, 최대 k)을 거리 종류별로 반환한다."""
        query = np.asarray(query)
        if query.ndim != 2 or query.shape[1] != self.n_features or not np.isfinite(query).all():
            raise ValueError("Invalid query features.")
        if self.device != "cpu" and self.l1_float32 and query.dtype != np.uint8:
            raise ValueError("uint8 GPU training requires uint8 queries.")
        labels = {m: np.empty((len(query), max(self.ks)), dtype=np.int64) for m in self.metrics}
        distances = {m: np.empty_like(a, dtype=np.float64) for m, a in labels.items()} if return_distances else {}
        updated = time.monotonic()
        # 배치 단위로 계산하므로 50,000×50,000 거리 행렬을 한꺼번에 만들지 않는다.
        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            batch = query[start:stop]
            result = self._cpu_neighbors(batch) if self.device == "cpu" else self._gpu_neighbors(batch)
            for metric, (indices, values) in result.items():
                labels[metric][start:stop] = self.y[indices]
                if return_distances:
                    distances[metric][start:stop] = values
            if time.monotonic() - updated >= 10:
                print(f"  neighbors: {stop}/{len(query)}", flush=True)
                updated = time.monotonic()
        return (labels, distances) if return_distances else labels

    def _cpu_neighbors(self, query):
        query = np.asarray(query, dtype=np.float64)
        result = {}
        if self.needs_dot:
            dot = query @ self.x64.T
            q_norm2 = np.einsum("ij,ij->i", query, query)
        for metric in self.metrics:
            if metric == "euclidean":
                # ||q-x||² = ||q||² + ||x||² - 2(q·x). 음수 반올림 오차는 0으로 보정한다.
                distance = np.maximum(q_norm2[:, None] + self.norm2[None, :] - 2 * dot, 0)
            elif metric == "cosine":
                # cosine 거리 = 1 - (q·x)/(||q|| ||x||). 0 벡터의 유사도는 0으로 둔다.
                denominator = np.sqrt(q_norm2)[:, None] * np.sqrt(self.norm2)[None, :]
                similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
                distance = 1 - np.clip(similarity, -1, 1)
            else:
                # Manhattan 거리 = 각 픽셀 차이의 절댓값을 더한 값.
                distance = cdist(query, self.x64, metric="cityblock")
            indices = np.argsort(distance, axis=1, kind="stable")[:, :max(self.ks)]
            values = np.take_along_axis(distance, indices, axis=1)
            # 순위는 제곱 L2로 구해도 같지만, 거리 가중 투표에는 실제 L2가 필요하다.
            result[metric] = (indices, np.sqrt(values) if metric == "euclidean" else values)
        return result

    def _gpu_neighbors(self, query):
        # CPU와 같은 공식이다. L2·cosine은 float64, CIFAR-10 L1은 float32를 쓴다.
        torch, result = self.torch, {}
        with torch.inference_mode():
            if self.needs_dot:
                q = torch.as_tensor(query, dtype=torch.float64, device=self.device)
                q_norm2, dot = (q * q).sum(1), q @ self.x64.T
            for metric in self.metrics:
                if metric == "euclidean":
                    distance = (q_norm2[:, None] + self.norm2[None, :] - 2 * dot).clamp_min_(0)
                elif metric == "cosine":
                    denominator = q_norm2.sqrt()[:, None] * self.norm2.sqrt()[None, :]
                    similarity = dot / denominator.clamp_min(torch.finfo(torch.float64).tiny)
                    distance = 1 - similarity.clamp(-1, 1)
                else:
                    q = torch.as_tensor(query, dtype=self.x_l1.dtype, device=self.device)
                    distance = torch.cdist(q, self.x_l1, p=1)
                # 동일 거리에서는 원래 학습 인덱스 순서를 유지해 결과를 재현한다.
                indices = torch.argsort(distance, dim=1, stable=True)[:, :max(self.ks)]
                values = torch.gather(distance, 1, indices)
                if metric == "euclidean":
                    values = values.sqrt()
                result[metric] = (indices.cpu().numpy(), values.cpu().numpy())
        return result


def vote(neighbor_labels, k_values, method="uniform_nearest", distances=None):
    """같은 거리순 이웃에 세 투표 규칙 중 하나를 적용해 k별 예측을 반환한다."""
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    ks = set(k_values)
    if labels.ndim != 2 or not ks or min(ks) < 1 or max(ks) > labels.shape[1]:
        raise ValueError("Invalid neighbors or k values.")
    if method not in VOTING or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Invalid voting method or class IDs.")
    weights = None
    if method == "distance_weighted":
        d = np.asarray(distances, dtype=np.float64)
        if d.shape != labels.shape or not np.isfinite(d).all() or np.any(d < 0) or np.any(np.diff(d, axis=1) < 0):
            raise ValueError("Supply matching finite nonnegative distances, sorted ascending.")
        # 거리 0인 이웃이 있으면 그 이웃들만 1표씩 투표한다. 나머지는 1/d 가중치다.
        zero = d == 0
        weights = np.divide(1.0, d, out=np.zeros_like(d), where=~zero)
        weights[zero.any(axis=1)] = zero[zero.any(axis=1)]
    votes = np.zeros((len(labels), 10), dtype=np.float64)
    rows, predictions = np.arange(len(labels)), {}
    for column in range(max(ks)):
        votes[rows, labels[:, column]] += 1 if weights is None else weights[:, column]
        k = column + 1
        if k not in ks:
            continue
        if method == "uniform_class_id":
            # argmax는 동률이면 첫 번째, 즉 가장 작은 클래스 번호를 고른다.
            predictions[k] = votes.argmax(axis=1)
        else:
            # 홀수 k라도 2:2:1처럼 동률이 생긴다. 최다 득표 클래스 중 최근접 이웃을 고른다.
            maximum = votes.max(axis=1, keepdims=True)
            winners = votes == maximum if weights is None else np.isclose(votes, maximum, rtol=1e-12, atol=0)
            positions = np.take_along_axis(winners, labels[:, :k], axis=1)
            predictions[k] = labels[rows, positions.argmax(axis=1)]
    return predictions


def evaluate(y_true, y_pred):
    # P/R/F1은 클래스별 지표를 같은 비중으로 평균한다(Macro). 미정의 지표는 0이다.
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(10), average="macro", zero_division=0,
    )
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                precision_macro=float(p), recall_macro=float(r), f1_macro=float(f1))


def write_csv(path, rows):
    # Excel에서도 열 이름을 읽기 쉽도록 UTF-8 BOM을 포함한다. 점수는 0~1로 저장한다.
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
