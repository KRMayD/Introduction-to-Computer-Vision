"""Exact, from-scratch KNN on every CIFAR-10 image; no package installation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import pickle
import platform
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits


METRICS = ("euclidean", "manhattan", "cosine")
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
DEFAULT_K = (1, 3, 5, 7, 9, 11, 15, 21)
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_cifar10(archive: Path):
    """Read only known members in memory; never extract or alter the archive.

    CIFAR's Python format uses pickle: only load a trusted dataset archive.
    """
    with tarfile.open(archive, "r:*") as handle:
        def read_batch(name):
            member = handle.getmember(f"cifar-10-batches-py/{name}")
            if not member.isfile():
                raise ValueError(f"Not a regular dataset file: {name}")
            if member.offset_data + member.size > archive.stat().st_size:
                raise ValueError("Dataset archive is incomplete; wait for its copy to finish.")
            with handle.extractfile(member) as stream:
                batch = pickle.load(stream, encoding="bytes")
            x = np.asarray(batch[b"data"])
            y = np.asarray(batch[b"labels"], dtype=np.int64)
            if x.shape != (10000, 3072) or y.shape != (10000,) or x.dtype != np.uint8:
                raise ValueError(f"Unexpected CIFAR-10 batch shape/type: {name}")
            return x, y

        batches = [read_batch(f"data_batch_{i}") for i in range(1, 6)]
        x_train = np.concatenate([x for x, _ in batches])
        y_train = np.concatenate([y for _, y in batches])
        x_test, y_test = read_batch("test_batch")
    if not np.array_equal(np.bincount(y_train, minlength=10), np.full(10, 5000)):
        raise ValueError("Training class counts do not match full CIFAR-10.")
    if not np.array_equal(np.bincount(y_test, minlength=10), np.full(10, 1000)):
        raise ValueError("Test class counts do not match full CIFAR-10.")
    return x_train, y_train, x_test, y_test


def majority_predictions(neighbor_labels, k_values, n_classes=10):
    """Uniform voting; a class vote tie selects the smallest class ID."""
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    if labels.ndim != 2 or max(k_values) > labels.shape[1]:
        raise ValueError("Not enough neighbors for the requested k values.")
    votes = np.zeros((len(labels), n_classes), dtype=np.int32)
    rows = np.arange(len(labels))
    requested = set(k_values)
    predictions = {}
    for column in range(max(k_values)):
        votes[rows, labels[:, column]] += 1
        if column + 1 in requested:
            predictions[column + 1] = votes.argmax(axis=1).copy()
    return predictions


def nearest_tie_predictions(neighbor_labels, k_values, n_classes=10):
    """Uniform voting; choose the nearest neighbor in a tied winning class."""
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    votes = np.zeros((len(labels), n_classes), dtype=np.int32)
    rows = np.arange(len(labels))
    predictions = {}
    for column in range(max(k_values)):
        votes[rows, labels[:, column]] += 1
        k = column + 1
        if k in k_values:
            winners = votes == votes.max(axis=1, keepdims=True)
            winning_positions = np.take_along_axis(winners, labels[:, :k], axis=1)
            predictions[k] = labels[rows, winning_positions.argmax(axis=1)].copy()
    return predictions


def distance_weighted_predictions(neighbor_labels, neighbor_distances, k_values, n_classes=10):
    """Inverse-distance voting with explicit zero-distance and tie handling.

    Neighbors must be ordered by nondecreasing distance. If any of the k
    distances is zero, only zero-distance neighbors vote, with equal weights.
    Equal total weights (relative tolerance 1e-12) are resolved by the closest
    neighbor belonging to a tied winning class, then stable neighbor order.
    """
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    if labels.ndim != 2 or distances.shape != labels.shape:
        raise ValueError("Neighbor labels and distances must have matching 2D shapes.")
    if not k_values or min(k_values) < 1 or max(k_values) > labels.shape[1]:
        raise ValueError("Invalid k values for the supplied neighbors.")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if np.any(np.diff(distances, axis=1) < 0):
        raise ValueError("Neighbors must be ordered by nondecreasing distance.")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("Neighbor class IDs are out of range.")
    zero = distances == 0
    weights = np.divide(1.0, distances, out=np.zeros_like(distances), where=~zero)
    has_zero = zero.any(axis=1)
    weights[has_zero] = zero[has_zero]
    votes = np.zeros((len(labels), n_classes), dtype=np.float64)
    rows = np.arange(len(labels))
    predictions = {}
    for column in range(max(k_values)):
        votes[rows, labels[:, column]] += weights[:, column]
        k = column + 1
        if k in k_values:
            winners = np.isclose(votes, votes.max(axis=1, keepdims=True), rtol=1e-12, atol=0)
            winning_positions = np.take_along_axis(winners, labels[:, :k], axis=1)
            predictions[k] = labels[rows, winning_positions.argmax(axis=1)].copy()
    return predictions


class ExactKNN:
    """Exhaustive neighbors using our own distances and voting.

    No sklearn neighbor estimator or approximate index is used. The GPU uses
    float64 dot products for Euclidean/cosine distances. For uint8 CIFAR pixels,
    L1 float32 sums are exact integers (< 2**24). Training-index order resolves
    equal-distance ties via stable sorting. Batching bounds working memory.
    """

    def __init__(self, k_values=DEFAULT_K, metrics=METRICS, device="cpu", batch_size=128):
        self.k_values = tuple(sorted(set(k_values)))
        self.metrics = tuple(metrics)
        self.device = device
        self.batch_size = batch_size
        if not self.k_values or min(self.k_values) < 1:
            raise ValueError("k values must be positive.")
        if not self.metrics or not set(self.metrics).issubset(METRICS):
            raise ValueError(f"Supported metrics: {METRICS}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

    def fit(self, x, y):
        x = np.asarray(x)
        self.y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or self.y.shape != (len(x),):
            raise ValueError("Expected a 2D feature array and one label per row.")
        if max(self.k_values) > len(x):
            raise ValueError("k exceeds the number of training examples.")
        if not np.isfinite(x).all():
            raise ValueError("Features must be finite.")
        self.n_features = x.shape[1]
        self.n_classes = int(self.y.max()) + 1
        self.needs_dot = bool({"euclidean", "cosine"}.intersection(self.metrics))
        self.l1_exact_float32 = x.dtype == np.uint8
        if self.device.startswith("cuda"):
            import torch
            self.torch = torch
            self.x64 = torch.as_tensor(x, dtype=torch.float64, device=self.device) if self.needs_dot else None
            self.norm2 = (self.x64 * self.x64).sum(1) if self.needs_dot else None
            self.norm = self.norm2.sqrt() if self.needs_dot else None
            l1_dtype = torch.float32 if self.l1_exact_float32 else torch.float64
            self.x_l1 = torch.as_tensor(x, dtype=l1_dtype, device=self.device) if "manhattan" in self.metrics else None
        else:
            self.x64 = np.asarray(x, dtype=np.float64)
            self.norm2 = np.einsum("ij,ij->i", self.x64, self.x64)
            self.norm = np.sqrt(self.norm2)
        return self

    def neighbor_labels(self, query, progress=None, return_distances=False):
        """Return sorted neighbor labels and, optionally, actual metric distances."""
        query = np.asarray(query)
        if query.ndim != 2 or query.shape[1] != self.n_features:
            raise ValueError("Query features do not match the training features.")
        if not np.isfinite(query).all():
            raise ValueError("Query features must be finite.")
        if self.device.startswith("cuda") and self.l1_exact_float32 and query.dtype != np.uint8:
            raise ValueError("A uint8 training array requires uint8 queries on the GPU.")
        max_k = max(self.k_values)
        labels = {metric: np.empty((len(query), max_k), dtype=np.int64) for metric in self.metrics}
        if return_distances:
            distances = {metric: np.empty((len(query), max_k), dtype=np.float64) for metric in self.metrics}
        last_update = time.perf_counter()
        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            if self.device.startswith("cuda"):
                batch_indices = self._gpu_indices(query[start:stop], max_k, return_distances)
            else:
                batch_indices = self._cpu_indices(query[start:stop], max_k, return_distances)
            for metric, result in batch_indices.items():
                if return_distances:
                    indices, batch_distances = result
                    distances[metric][start:stop] = batch_distances
                else:
                    indices = result
                labels[metric][start:stop] = self.y[indices]
            if progress and (time.perf_counter() - last_update >= 10 or stop == len(query)):
                progress(stop, len(query))
                last_update = time.perf_counter()
        return (labels, distances) if return_distances else labels

    def _gpu_indices(self, query, max_k, return_distances=False):
        torch = self.torch
        result = {}
        with torch.inference_mode():
            if self.needs_dot:
                q64 = torch.as_tensor(query, dtype=torch.float64, device=self.device)
                q_norm2 = (q64 * q64).sum(1)
                dot = q64 @ self.x64.T
            for metric in self.metrics:
                if metric == "euclidean":
                    # sqrt is monotone, so squared L2 has identical neighbors.
                    distance = (q_norm2[:, None] + self.norm2[None, :] - 2 * dot).clamp_min_(0)
                elif metric == "cosine":
                    denominator = q_norm2.sqrt()[:, None] * self.norm[None, :]
                    similarity = dot / denominator.clamp_min(torch.finfo(torch.float64).tiny)
                    distance = (1 - similarity.clamp(-1, 1)).clamp_min_(0)
                else:
                    q_l1 = torch.as_tensor(query, dtype=self.x_l1.dtype, device=self.device)
                    distance = torch.cdist(q_l1, self.x_l1, p=1)
                indices = torch.argsort(distance, dim=1, stable=True)[:, :max_k]
                if return_distances:
                    selected_distances = torch.gather(distance, 1, indices)
                    if metric == "euclidean":
                        selected_distances = selected_distances.sqrt()
                    result[metric] = (indices.cpu().numpy(), selected_distances.cpu().numpy())
                else:
                    result[metric] = indices.cpu().numpy()
        return result

    def _cpu_indices(self, query, max_k, return_distances=False):
        from scipy.spatial.distance import cdist
        query = np.asarray(query, dtype=np.float64)
        result = {}
        if self.needs_dot:
            dot = query @ self.x64.T
            q_norm2 = np.einsum("ij,ij->i", query, query)
        for metric in self.metrics:
            if metric == "euclidean":
                distance = np.maximum(q_norm2[:, None] + self.norm2[None, :] - 2 * dot, 0)
            elif metric == "cosine":
                denominator = np.sqrt(q_norm2)[:, None] * self.norm[None, :]
                similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
                distance = 1 - np.clip(similarity, -1, 1)
            else:
                distance = cdist(query, self.x64, metric="cityblock")
            indices = np.argsort(distance, axis=1, kind="stable")[:, :max_k]
            if return_distances:
                selected_distances = np.take_along_axis(distance, indices, axis=1)
                if metric == "euclidean":
                    selected_distances = np.sqrt(selected_distances)
                result[metric] = (indices, selected_distances)
            else:
                result[metric] = indices
        return result

    def predict_all(self, query, progress=None):
        neighbors = self.neighbor_labels(query, progress)
        return {
            metric: majority_predictions(labels, self.k_values, self.n_classes)
            for metric, labels in neighbors.items()
        }


def evaluate(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(10), average="macro", zero_division=0,
    )
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                precision_macro=float(precision), recall_macro=float(recall), f1_macro=float(f1))


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(requested, threads):
    if requested == "cpu":
        return "cpu", "CPU"
    try:
        import torch
    except ImportError:
        if requested != "auto":
            raise RuntimeError("PyTorch is unavailable; use --device cpu.")
        return "cpu", "CPU"
    torch.set_num_threads(threads)
    if requested == "auto" and not torch.cuda.is_available():
        return "cpu", "CPU"
    device = "cuda:0" if requested == "auto" else requested
    # CUDA indices are logical indices and respect CUDA_VISIBLE_DEVICES.
    return device, torch.cuda.get_device_name(torch.device(device))


def make_plots(output, summary, test_results, best, y_test, test_predictions):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"euclidean": "#2563eb", "manhattan": "#dc2626", "cosine": "#059669"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, score in zip(axes.flat, SCORES):
        for metric in METRICS:
            rows = [row for row in summary if row["metric"] == metric]
            ax.errorbar([r["k"] for r in rows], [r[f"{score}_mean"] for r in rows],
                        yerr=[r[f"{score}_std"] for r in rows], marker="o", capsize=3,
                        label=metric, color=colors[metric])
        ax.set(xlabel="k", ylabel=score, title=f"5-fold CV: {score}")
        ax.legend()
    fig.savefig(output / "cross_validation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(SCORES))
    for i, row in enumerate(test_results):
        ax.bar(positions + (i - 1) * 0.25, [row[s] for s in SCORES], width=0.24,
               label=f"{row['metric']} (k={row['k']})", color=colors[row["metric"]])
    ax.set_xticks(positions, SCORES)
    ax.set(ylabel="Score", ylim=(0, 1), title="Official test set: 10,000 images")
    ax.legend()
    fig.savefig(output / "test_metrics.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(y_test, test_predictions[best["metric"]], labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    picture = ax.imshow(matrix, cmap="Blues")
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title=f"CV-selected model: {best['metric']}, k={best['k']}")
    fig.colorbar(picture, ax=ax)
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    write_csv(output / "confusion_matrix.csv", [dict(actual=name, **dict(zip(CLASS_NAMES, map(int, matrix[i]))))
                                               for i, name in enumerate(CLASS_NAMES)])


def make_report(output, metadata, summary, test_results, best, per_class):
    best_test = next(row for row in test_results if row["selected_by_cv"])
    def table(rows, columns, percentage_columns=()):
        cells = ["<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"]
        for row in rows:
            values = [f"{row[c] * 100:.2f}%" if c in percentage_columns else str(row[c]) for c in columns]
            cells.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
        return "\n".join(cells) + "</tbody></table>"

    cv_rows = []
    for row in summary:
        display = {"metric": row["metric"], "k": row["k"]}
        display.update({score: f"{row[f'{score}_mean'] * 100:.2f} ± {row[f'{score}_std'] * 100:.2f}%" for score in SCORES})
        cv_rows.append(display)
    class_rows = [dict(class_name=name, **{key: per_class[name][key] for key in ("precision", "recall", "f1-score", "support")})
                  for name in CLASS_NAMES]
    metric_winners = {}
    for metric in METRICS:
        candidates = [r for r in summary if r["metric"] == metric]
        metric_winners[metric] = min(candidates, key=lambda r: (-r["accuracy_mean"], r["k"]))
    winner_text = "; ".join(f"{metric}: k={row['k']}, CV 정확도 {row['accuracy_mean']:.2%}"
                            for metric, row in metric_winners.items())
    k1 = next(r for r in summary if r["metric"] == best["metric"] and r["k"] == 1)
    k_effect = (best["accuracy_mean"] - k1["accuracy_mean"]) * 100
    best_classes = sorted(CLASS_NAMES, key=lambda name: per_class[name]["f1-score"], reverse=True)
    content = f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIFAR-10 전체 데이터 KNN 실험 보고서</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;line-height:1.7;color:#172033}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;margin:20px 0;font-size:14px}}td,th{{border:1px solid #d9dee8;padding:8px 12px;text-align:right}}th{{background:#edf2f8}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}code,pre{{background:#edf2f8;padding:3px 6px}}.result{{background:#e9f3ff;padding:16px;border-radius:8px}}a{{color:#185bb5}}</style>
<h1>CIFAR-10 전체 데이터 KNN 실험 보고서</h1>
<p class="result">5-fold CV로 선택한 모델: <strong>{best['metric']}, k={best['k']}</strong>.<br>
공식 테스트 10,000장: Accuracy <strong>{best_test['accuracy']:.2%}</strong>,
Macro Precision {best_test['precision_macro']:.2%}, Macro Recall {best_test['recall_macro']:.2%},
Macro F1 {best_test['f1_macro']:.2%}.</p>
<h2>1. 데이터와 실험 설정</h2>
<p>사용자가 제공한 <code>{html.escape(metadata['archive'])}</code>의 학습 배치 5개, 총 <strong>50,000장 전체</strong>와
공식 테스트 배치 <strong>10,000장 전체</strong>를 사용했다. 클래스는 10개이며 학습은 클래스당 5,000장, 테스트는 클래스당 1,000장이다.
32×32 RGB 이미지를 3,072차원 벡터로 펼쳤다. 원본 uint8 픽셀 0~255를 사용했고, 샘플링·리사이징·PCA·특징 추출은 사용하지 않았다.
세 거리 모두 전체 픽셀에 동일 상수를 곱해도 이웃 순서가 같으므로 /255 정규화는 생략했다.</p>
<p>k = {list(metadata['k_values'])}. StratifiedKFold(n_splits=5, shuffle=True, random_state={metadata['seed']})로
각 fold에서 <strong>40,000장을 학습하고 10,000장을 검증</strong>했다. 모든 학습 이미지는 정확히 한 번 검증에 사용된다.
모든 설정은 동일한 fold를 사용한다. 테스트 데이터는 교차검증과 모델 선택에 사용하지 않았다.</p>
<h2>2. KNN 직접 구현</h2>
<p>KNN은 학습 단계에서 특징과 정답을 저장하고, 예측 단계에서 모든 학습 이미지까지의 거리를 계산한 뒤 가까운 k개의 레이블을 다수결로 분류한다.
거리는 직접 계산하고, 이웃 선택에는 안정 정렬, 분류에는 누적 투표를 구현했다. sklearn의 KNeighborsClassifier는 본 실험에 사용하지 않았다.
scikit-learn은 fold 분할과 평가 지표 계산에 사용한다.</p>
<ul><li>Euclidean: d(x,z)=√Σ(xⱼ−zⱼ)². 순위가 동일한 제곱 거리를 계산한다.</li>
<li>Manhattan: d(x,z)=Σ|xⱼ−zⱼ|.</li><li>Cosine: d(x,z)=1−(x·z)/(‖x‖‖z‖). 영벡터가 포함되면 유사도를 0으로 정의한다.</li></ul>
<p>모든 학습점과의 거리를 계산하는 정확 탐색이며 근사 탐색을 사용하지 않았다. 거리 동률은 학습 배열의 앞선 인덱스,
투표 동률은 작은 클래스 ID를 선택한다. 단일 GPU({html.escape(metadata['device_name'])}, {metadata['device']})에서
배치 {metadata['batch_size']}로 계산해 전체 거리 행렬을 저장하지 않았다.
Euclidean·Cosine은 float64, Manhattan은 정수 픽셀의 합이 정확히 표현되는 float32를 사용했다.
Euclidean과 Cosine의 내적을 공유하고, 한 번 구한 이웃으로 모든 k를 평가했다.</p>
<h2>3. 평가 지표와 모델 선택 기준</h2>
<p>Accuracy는 전체 정답 비율이다. 클래스별 Precision=TP/(TP+FP), Recall=TP/(TP+FN),
F1=2PR/(P+R)을 계산하고 10개 클래스의 값을 동일 가중 평균한 macro 지표를 보고한다.
Macro F1은 클래스별 F1의 평균이며 macro precision·recall의 조화평균이 아니다. 분모가 0이면 해당 지표는 0이다.</p>
<p>최적 설정은 <strong>평균 CV Accuracy</strong>가 가장 높은 조합이다. 동률이면 작은 k, 그다음 거리 이름의 사전순으로 선택한다.
거리별 최적 k도 같은 기준으로 선택하여 최종적으로 50,000장 전체를 저장한 모델을 공식 테스트 10,000장에 평가했다.
아래 표는 fold별 지표의 평균 ± 표본 표준편차(ddof=1)이며 신뢰구간이 아니다.</p>
{table(cv_rows, ['metric', 'k', *SCORES])}
<img src="cross_validation.png" alt="k와 거리별 교차검증 평균 및 표준편차">
<h2>4. 공식 테스트 결과</h2>
{table(test_results, ['metric', 'k', 'selected_by_cv', *SCORES], SCORES)}
<p>selected_by_cv=True인 모델은 테스트 결과를 보기 전에 전체 CV 표에서 선택한 최종 모델이다.
다른 행은 각 거리에서 CV로 선택한 k의 비교 결과다. 테스트 점수로 k나 최종 모델을 다시 선택하지 않았다.</p>
<img src="test_metrics.png" alt="거리별 최적 k의 테스트 평가 지표">
<h2>5. 비교 및 설명</h2>
<p>거리별 최적 결과는 {winner_text}이다. 최종 선택 거리({best['metric']})에서 최적 k는 k=1 대비 CV Accuracy가
{k_effect:+.2f} 퍼센트포인트 차이 났다. k가 커지면 한 개 이웃의 잡음 영향은 줄지만 다른 클래스의 표본이 투표에 참여할 수 있어,
k를 크게 하는 것이 항상 성능 향상으로 이어지지는 않는다.</p>
<p>Euclidean은 큰 픽셀 차이의 영향을 제곱으로 반영하고, Manhattan은 절댓값 차이를 선형으로 더한다.
Cosine은 벡터의 크기보다 방향을 비교한다. 어느 거리가 유리한지는 이미지의 픽셀 분포에 달려 있으며 위 CV 결과로 비교했다.
원시 픽셀 KNN은 객체 위치·배경·조명 변화에 민감하고, 픽셀 거리가 의미상 유사성을 충분히 표현하지 못한다는 한계가 있다.</p>
<p>선택 모델의 테스트 클래스별 F1이 높은 클래스는 {best_classes[0]}({per_class[best_classes[0]]['f1-score']:.2%}),
{best_classes[1]}({per_class[best_classes[1]]['f1-score']:.2%})이고,
낮은 클래스는 {best_classes[-1]}({per_class[best_classes[-1]]['f1-score']:.2%}),
{best_classes[-2]}({per_class[best_classes[-2]]['f1-score']:.2%})이다.
테스트 데이터는 클래스별 표본 수가 같으므로 Accuracy와 Macro Recall이 일치한다.
5개 fold는 학습 데이터가 겹치므로 작은 점수 차이를 독립적인 통계적 유의성으로 해석하지 않았다.</p>
<img src="confusion_matrix.png" alt="선택 모델의 테스트 혼동 행렬">
{table(class_rows, ['class_name', 'precision', 'recall', 'f1-score', 'support'], ['precision', 'recall', 'f1-score'])}
<h2>6. 재현 및 산출물</h2>
<p>실험 시작: {metadata['started_at']}. 실행 시간: {metadata['elapsed_seconds']:.1f}초.
Python {metadata['python']}, NumPy {metadata['numpy']}, scikit-learn {metadata['sklearn']}, PyTorch {metadata.get('torch', 'N/A')}.</p>
<p>실행 명령: <code>{html.escape(metadata['command'])}</code><br>
입력 SHA-256: <code>{metadata['archive_sha256']}</code></p>
<ul><li><a href="cv_folds.csv">fold별 모든 지표</a>, <a href="cv_summary.csv">CV 평균·표준편차 및 전체 OOF 지표</a></li>
<li><a href="test_results.csv">테스트 지표</a>, <a href="classification_report.csv">클래스별 지표</a>, <a href="confusion_matrix.csv">혼동 행렬</a></li>
<li><a href="fold_assignments.csv">학습 이미지별 검증 fold</a>, <a href="predictions.npz">OOF·테스트 예측 및 정답</a></li>
<li><a href="metadata.json">설정·환경·시간·모델 선택 정보</a></li></ul>
</html>"""
    (output / "report.html").write_text(content, encoding="utf-8")


def run(args):
    import sklearn
    start = time.perf_counter()
    started_at = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    archive = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else archive.parent / "results" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("full_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    device, device_name = resolve_device(args.device, args.threads)
    log_stream = (output / "run.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        text = f"[{time.perf_counter() - start:7.1f}s] {message}"
        print(text, flush=True)
        log_stream.write(text + "\n")

    log(f"Output: {output}")
    log(f"Device: {device} ({device_name}); CPU threads: {args.threads}")
    x_train, y_train, x_test, y_test = load_cifar10(archive)
    log(f"FULL DATA: train={x_train.shape}, test={x_test.shape}; k={args.k_values}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    archive_hash = digest.hexdigest()
    metadata = dict(archive=str(archive), archive_sha256=archive_hash,
                    started_at=started_at, seed=args.seed, k_values=args.k_values,
                    metrics=list(METRICS), folds=5, train_images=len(y_train), test_images=len(y_test),
                    features=x_train.shape[1], train_class_counts=np.bincount(y_train).tolist(),
                    test_class_counts=np.bincount(y_test).tolist(), device=device, device_name=device_name,
                    batch_size=args.batch_size, threads=args.threads, python=platform.python_version(),
                    numpy=np.__version__, sklearn=sklearn.__version__,
                    selection_rule="highest mean CV accuracy; then smallest k; then metric name",
                    command=" ".join([sys.executable, *sys.argv]))
    if device.startswith("cuda"):
        import torch
        metadata["torch"] = torch.__version__
        metadata["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    oof = {(metric, k): np.full(len(y_train), -1, dtype=np.int64) for metric in METRICS for k in args.k_values}
    fold_assignment = np.zeros(len(y_train), dtype=np.int8)
    rows = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_index, valid_index) in enumerate(splitter.split(x_train, y_train), 1):
        log(f"Fold {fold}/5: train={len(train_index)}, validation={len(valid_index)}")
        fold_assignment[valid_index] = fold
        classifier = ExactKNN(args.k_values, device=device, batch_size=args.batch_size).fit(x_train[train_index], y_train[train_index])
        predictions = classifier.predict_all(x_train[valid_index], lambda n, total: log(f"Fold {fold}: {n}/{total} queries"))
        for metric in METRICS:
            for k in args.k_values:
                predicted = predictions[metric][k]
                oof[metric, k][valid_index] = predicted
                rows.append(dict(fold=fold, metric=metric, k=k, n_train=len(train_index),
                                 n_valid=len(valid_index), **evaluate(y_train[valid_index], predicted)))
        del classifier
        write_csv(output / "cv_folds.csv", rows)
        log(f"Fold {fold}/5 complete")
    if np.any(fold_assignment == 0) or any(np.any(pred < 0) for pred in oof.values()):
        raise RuntimeError("Incomplete out-of-fold coverage.")

    summary = []
    for metric in METRICS:
        for k in args.k_values:
            group = [row for row in rows if row["metric"] == metric and row["k"] == k]
            record = dict(metric=metric, k=k)
            for score in SCORES:
                values = [row[score] for row in group]
                record[f"{score}_mean"] = float(np.mean(values))
                record[f"{score}_std"] = float(np.std(values, ddof=1))
            record.update({f"{score}_oof": value for score, value in evaluate(y_train, oof[metric, k]).items()})
            summary.append(record)
    write_csv(output / "cv_summary.csv", summary)
    write_csv(output / "fold_assignments.csv", (dict(train_index=i, label=int(y_train[i]), validation_fold=int(fold_assignment[i])) for i in range(len(y_train))))

    selection_key = lambda row: (-row["accuracy_mean"], row["k"], row["metric"])
    best = min(summary, key=selection_key)
    winners = {metric: min([row for row in summary if row["metric"] == metric], key=selection_key) for metric in METRICS}
    metadata.update(selected_model=best, per_metric_selected_k={metric: row["k"] for metric, row in winners.items()},
                    cv_elapsed_seconds=time.perf_counter() - start)
    # Persist selection before computing any test predictions.
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Selected using CV ONLY: {best['metric']}, k={best['k']}, accuracy={best['accuracy_mean']:.6f}")
    selected_ks = tuple(sorted({row["k"] for row in winners.values()}))
    classifier = ExactKNN(selected_ks, device=device, batch_size=args.batch_size).fit(x_train, y_train)
    final_all = classifier.predict_all(x_test, lambda n, total: log(f"Full-train official test: {n}/{total} queries"))
    del classifier
    test_results, test_predictions = [], {}
    for metric in METRICS:
        k = winners[metric]["k"]
        test_predictions[metric] = final_all[metric][k]
        row = dict(metric=metric, k=k, selected_by_cv=(metric == best["metric"]),
                   n_train=len(y_train), n_test=len(y_test), **evaluate(y_test, test_predictions[metric]))
        test_results.append(row)
        log(f"TEST {metric}, k={k}: " + ", ".join(f"{score}={row[score]:.6f}" for score in SCORES))
    write_csv(output / "test_results.csv", test_results)
    per_class = classification_report(y_test, test_predictions[best["metric"]], labels=np.arange(10),
                                      target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    write_csv(output / "classification_report.csv", [dict(class_name=name, **per_class[name]) for name in CLASS_NAMES])
    np.savez_compressed(output / "predictions.npz", y_train=y_train, y_test=y_test, validation_fold=fold_assignment,
                        **{f"oof_{metric}_k{k}": pred for (metric, k), pred in oof.items()},
                        **{f"test_{metric}_k{winners[metric]['k']}": pred for metric, pred in test_predictions.items()})
    make_plots(output, summary, test_results, best, y_test, test_predictions)
    metadata["elapsed_seconds"] = time.perf_counter() - start
    metadata["test_results"] = test_results
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(output, metadata, summary, test_results, best, per_class)
    log(f"Complete in {metadata['elapsed_seconds']:.1f}s. Report: {output / 'report.html'}")
    log_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "cifar-10-python.tar")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist.")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a logical CUDA device such as cuda:0")
    args = parser.parse_args()
    args.k_values = sorted(set(args.k_values))
    if not args.k_values or min(args.k_values) < 1 or max(args.k_values) > 40000:
        parser.error("k must be between 1 and 40,000 (the fold training size).")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive.")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
