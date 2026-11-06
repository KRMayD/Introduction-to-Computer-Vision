"""Full CIFAR-10 5-fold experiments with configurable KNN voting rules and k."""

# 03(세 투표 방식 비교)과 04(k=101 확장)가 함께 사용하는 실제 실험 실행부.
# 기본 실험의 데이터·fold·예측을 대조한 뒤, 동일한 이웃에 서로 다른 투표 규칙을 적용한다.
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from knn_cifar10.core import (
    CLASS_NAMES, METRICS, SCORES, ExactKNN, distance_weighted_predictions,
    evaluate, load_cifar10, majority_predictions, nearest_tie_predictions,
    resolve_device, write_csv,
)


VOTING = ("uniform_class_id", "uniform_nearest", "distance_weighted")


def all_predictions(labels, distances, ks, voting_methods=VOTING):
    # labels는 거리순 이웃 클래스, distances는 그 실제 거리다.
    # 이웃 탐색은 공유하고 선택한 투표 함수만 호출해 비교 조건을 같게 유지한다.
    predictions = {}
    if "uniform_class_id" in voting_methods:
        predictions["uniform_class_id"] = majority_predictions(labels, ks)
    if "uniform_nearest" in voting_methods:
        predictions["uniform_nearest"] = nearest_tie_predictions(labels, ks)
    if "distance_weighted" in voting_methods:
        predictions["distance_weighted"] = distance_weighted_predictions(labels, distances, ks)
    return predictions


def run(args):
    # 1. 기본 실험 설정을 읽고 이번에 비교할 투표 방식·k 범위를 결정한다.
    started = time.perf_counter()
    source = args.baseline.resolve()
    baseline_meta = json.loads((source / "metadata.json").read_text())
    voting_methods = tuple(args.voting_methods)
    # 다수결은 이웃의 순서와 클래스만 필요하다. 가중 투표가 있을 때만 거리 배열도 저장한다.
    use_distances = "distance_weighted" in voting_methods
    prefix = f"nearest_k{args.max_k or max(baseline_meta['k_values'])}" if voting_methods == ("uniform_nearest",) else "weighted_full"
    stamp = prefix + datetime.now(ZoneInfo("Asia/Seoul")).strftime("_%Y%m%d_%H%M%S")
    output = args.output.resolve() if args.output else source.parent / stamp
    # 새 출력 디렉토리만 허용해 이미 수행한 실험 결과를 덮어쓰지 않는다.
    output.mkdir(parents=True, exist_ok=False)
    log_file = (output / "run.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        line = f"[{time.perf_counter()-started:7.1f}s] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Output: {output}")
    device, device_name = resolve_device(args.device, args.threads)
    archive = args.data.expanduser().resolve() if args.data else Path(baseline_meta["archive"])
    x, y, test_x, test_y = load_cifar10(archive)
    # 2. 전체 데이터를 로드하고 기본 실험과 같은 파일인지 SHA-256으로 확인한다.
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != baseline_meta["archive_sha256"]:
        raise ValueError("Dataset hash differs from the baseline experiment.")
    # --max-k 101이면 1,3,...,101의 51개 값, 생략하면 기본 실험의 8개 값을 사용한다.
    ks = list(range(1, args.max_k + 1, 2)) if args.max_k else baseline_meta["k_values"]
    shared_ks = [k for k in baseline_meta["k_values"] if k in ks]
    seed = baseline_meta["seed"]
    batch_size = baseline_meta["batch_size"]
    # 이전 OOF 예측과 fold 번호를 읽어 단순히 점수만 비슷한 것이 아니라 예측까지 대조한다.
    with np.load(source / "predictions.npz") as original:
        np.testing.assert_array_equal(original["y_train"], y)
        np.testing.assert_array_equal(original["y_test"], test_y)
        baseline_folds = original["validation_fold"].copy()
        baseline_oof = {(metric, k): original[f"oof_{metric}_k{k}"].copy() for metric in METRICS for k in shared_ks}
        # 이 과제의 기본 실험은 세 거리 모두 k=1이 선택되어 해당 키로 저장되어 있다.
        baseline_test = {metric: original[f"test_{metric}_k1"].copy() for metric in METRICS}
    metadata = dict(archive=str(archive), archive_sha256=digest.hexdigest(), baseline_results=str(source),
                    started_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    train_images=len(y), test_images=len(test_y), features=x.shape[1],
                    folds=5, seed=seed, k_values=ks, metrics=list(METRICS), voting_methods=list(voting_methods),
                    weighting="1 / actual metric distance; zero distances exclusively receive equal votes" if use_distances else "uniform: one vote per neighbor",
                    weighted_score_tie="closest neighbor in a tied winning class; rtol=1e-12, atol=0" if use_distances else None,
                    uniform_vote_tie="closest neighbor in a tied winning class" if "uniform_nearest" in voting_methods else "smallest class ID",
                    neighbor_distance_tie="stable training-index order",
                    selection_rule="highest mean CV accuracy, then smallest k, then metric name",
                    device=device, device_name=device_name, threads=args.threads, batch_size=batch_size,
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    command=" ".join([sys.executable, *sys.argv]))
    if device.startswith("cuda"):
        import torch
        metadata["torch"] = torch.__version__

    def save_metadata():
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_metadata()
    log(f"FULL DATA train={x.shape}, test={test_x.shape}; device={device}; 5-fold; k={ks}")
    # 3. OOF 예측은 (투표 방식, 거리, k)별로 이미지 50,000개의 1차원 배열을 갖는다.
    # 이웃 캐시는 (50,000, 최대 k) 크기다. 이미지 전체 거리 행렬을 저장하는 것은 아니다.
    oof = {(vote, metric, k): np.full(len(y), -1, dtype=np.int8) for vote in voting_methods for metric in METRICS for k in ks}
    neighbor_labels = {metric: np.empty((len(y), max(ks)), dtype=np.int8) for metric in METRICS}
    neighbor_distances = {metric: np.empty((len(y), max(ks)), dtype=np.float64) for metric in METRICS} if use_distances else {}
    assignment = np.zeros(len(y), dtype=np.int8)
    fold_rows = []
    for fold, (training, validation) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y), 1):
        # 기본 실험과 검증 이미지 인덱스가 완전히 같은지 확인해 공정하게 비교한다.
        np.testing.assert_array_equal(np.flatnonzero(baseline_folds == fold), validation)
        assignment[validation] = fold
        log(f"Fold {fold}/5: train={len(training)}, validation={len(validation)}")
        # 현재 fold의 학습 40,000장만 fit에 사용한다. 검증 10,000장은 참조 데이터에서 제외된다.
        classifier = ExactKNN(ks, device=device, batch_size=batch_size).fit(x[training], y[training])
        result = classifier.neighbor_labels(x[validation], lambda n, total: log(f"Fold {fold}: {n}/{total} queries"), return_distances=use_distances)
        labels, distances = result if use_distances else (result, {})
        del classifier
        for metric in METRICS:
            neighbor_labels[metric][validation] = labels[metric]
            if use_distances:
                neighbor_distances[metric][validation] = distances[metric]
            # k 범위를 확장했어도 기존 k의 기본 투표 예측이 유지되는지 검사한다.
            if shared_ks:
                original_predictions = majority_predictions(labels[metric], shared_ks)
                for k in shared_ks:
                    np.testing.assert_array_equal(original_predictions[k], baseline_oof[metric, k][validation])
            # 가장 비싼 이웃 탐색은 한 번 수행하고, 그 결과를 모든 투표 방식·k에 재사용한다.
            predictions = all_predictions(labels[metric], distances.get(metric), ks, voting_methods)
            for vote in voting_methods:
                for k in ks:
                    pred = predictions[vote][k]
                    if vote == "uniform_class_id" and k in shared_ks:
                        np.testing.assert_array_equal(pred, baseline_oof[metric, k][validation])
                    # 검증 예측을 원래 이미지 위치에 넣고 fold별 4개 평가 지표를 기록한다.
                    oof[vote, metric, k][validation] = pred
                    fold_rows.append(dict(voting=vote, metric=metric, k=k, fold=fold,
                                          n_train=len(training), n_valid=len(validation), **evaluate(y[validation], pred)))
        write_csv(output / "cv_folds.csv", fold_rows)
        log(f"Fold {fold}/5 complete; original uniform predictions reproduced exactly")
    assert np.all(assignment > 0) and all(np.all(pred >= 0) for pred in oof.values())
    # 4. 각 설정의 5개 fold 지표로 평균 ± 표본 표준편차를 계산한다.
    summary = []
    for vote in voting_methods:
        for metric in METRICS:
            for k in ks:
                group = [r for r in fold_rows if r["voting"] == vote and r["metric"] == metric and r["k"] == k]
                row = dict(voting=vote, metric=metric, k=k)
                for score in SCORES:
                    values = [r[score] for r in group]
                    row[score+"_mean"] = float(np.mean(values))
                    row[score+"_std"] = float(np.std(values, ddof=1))
                # 통합 OOF 점수와 fold 평균은 별개다. 결과 표의 기본 값은 fold 평균이다.
                row.update({score+"_oof": value for score, value in evaluate(y, oof[vote, metric, k]).items()})
                summary.append(row)
    write_csv(output / "cv_summary.csv", summary)
    if use_distances:
        write_csv(output / "weighted_cv_summary.csv", [r for r in summary if r["voting"] == "distance_weighted"])
    write_csv(output / "fold_assignments.csv", [dict(train_index=i, label=int(y[i]), validation_fold=int(assignment[i])) for i in range(len(y))])
    np.savez_compressed(output / "oof_neighbors.npz", validation_fold=assignment, y_train=y,
                        **{f"{m}_labels": a for m, a in neighbor_labels.items()},
                        **{f"{m}_distances": a for m, a in neighbor_distances.items()})
    # 5. 테스트 점수 없이 CV 평균 정확도만으로 각 투표 방식의 최종 모델을 선택한다.
    # selected는 방식별 최종 거리·k, per_metric은 방식·거리별 최적 k다.
    key = lambda r: (-r["accuracy_mean"], r["k"], r["metric"])
    selected = {vote: min([r for r in summary if r["voting"] == vote], key=key) for vote in voting_methods}
    per_metric = {(vote, metric): min([r for r in summary if r["voting"] == vote and r["metric"] == metric], key=key)
                  for vote in voting_methods for metric in METRICS}
    metadata.update(selected_by_voting=selected,
                    per_metric_selected_k={vote: {m: per_metric[vote, m]["k"] for m in METRICS} for vote in voting_methods},
                    original_oof_predictions_reproduced=True, original_fold_assignments_reproduced=True,
                    verified_original_k_values=shared_ks,
                    cv_elapsed_seconds=time.perf_counter()-started)
    save_metadata()  # Selection is persisted before test prediction.
    for vote, row in selected.items():
        log(f"CV SELECTED {vote}: {row['metric']}, k={row['k']}, accuracy={row['accuracy_mean']:.6f}")
    # 6. 선택 결과 저장 후 학습 50,000장을 모두 사용해 공식 테스트 10,000장을 예측한다.
    # 필요한 최적 k들의 합집합만 사용하므로 테스트에서는 전체 k 탐색을 반복하지 않는다.
    final_ks = sorted({r["k"] for r in per_metric.values()})
    classifier = ExactKNN(final_ks, device=device, batch_size=batch_size).fit(x, y)
    result = classifier.neighbor_labels(test_x, lambda n, total: log(f"Full-train test: {n}/{total} queries"), return_distances=use_distances)
    test_labels, test_distances = result if use_distances else (result, {})
    del classifier
    test_predictions, test_rows = {}, []
    for metric in METRICS:
        # 첫 이웃은 k=1 예측이므로 기본 실험의 테스트 예측과 정확히 같아야 한다.
        np.testing.assert_array_equal(test_labels[metric][:, 0], baseline_test[metric])
        predictions = all_predictions(test_labels[metric], test_distances.get(metric), final_ks, voting_methods)
        for vote in voting_methods:
            k = per_metric[vote, metric]["k"]
            pred = predictions[vote][k]
            test_predictions[vote, metric, k] = pred
            if vote == "uniform_class_id" and k == 1:
                np.testing.assert_array_equal(pred, baseline_test[metric])
            row = dict(voting=vote, metric=metric, k=k, selected_within_voting=(selected[vote]["metric"] == metric),
                       n_train=len(y), n_test=len(test_y), **evaluate(test_y, pred))
            test_rows.append(row)
            log(f"TEST {vote} {metric} k={k}: " + ", ".join(f"{s}={row[s]:.6f}" for s in SCORES))
    write_csv(output / "test_results.csv", test_rows)
    if use_distances:
        write_csv(output / "weighted_test_results.csv", [r for r in test_rows if r["voting"] == "distance_weighted"])
    write_csv(output / "voting_comparison.csv", [dict(voting=vote, metric=selected[vote]["metric"], k=selected[vote]["k"],
              cv_accuracy=selected[vote]["accuracy_mean"], cv_accuracy_std=selected[vote]["accuracy_std"],
              **evaluate(test_y, test_predictions[vote, selected[vote]["metric"], selected[vote]["k"]])) for vote in voting_methods])
    # 7. 각 투표 방식의 최종 모델에 대해 클래스별 지표·혼동행렬을 추가로 저장한다.
    class_rows = []
    for vote in voting_methods:
        metric, k = selected[vote]["metric"], selected[vote]["k"]
        pred = test_predictions[vote, metric, k]
        report = classification_report(test_y, pred, labels=np.arange(10), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        class_rows.extend(dict(voting=vote, metric=metric, k=k, class_name=name, **report[name]) for name in CLASS_NAMES)
        matrix = confusion_matrix(test_y, pred, labels=np.arange(10))
        write_csv(output / f"confusion_{vote}.csv", [dict(actual=name, **dict(zip(CLASS_NAMES, map(int, matrix[i])))) for i, name in enumerate(CLASS_NAMES)])
    write_csv(output / "classification_report.csv", class_rows)
    # CSV는 점수를, NPZ는 실제 예측·정답·이웃을 보관해 나중에 결과를 검증할 수 있게 한다.
    np.savez_compressed(output / "predictions.npz", y_train=y, y_test=test_y, validation_fold=assignment,
                        **{f"oof_{vote}_{metric}_k{k}": pred for (vote, metric, k), pred in oof.items()},
                        **{f"test_{vote}_{metric}_k{k}": pred for (vote, metric, k), pred in test_predictions.items()})
    np.savez_compressed(output / "test_neighbors.npz", y_test=test_y,
                        **{f"{m}_labels": a.astype(np.int8) for m, a in test_labels.items()},
                        **{f"{m}_distances": a for m, a in test_distances.items()})
    metadata.update(test_results=test_rows, original_test_predictions_reproduced=True,
                    elapsed_seconds=time.perf_counter()-started)
    save_metadata()
    log(f"COMPLETE in {metadata['elapsed_seconds']:.1f}s; results: {output}")
    log_file.close()


def main():
    # 실행 진입점에서 넘긴 옵션을 읽고 범위를 검증한다. CPU 스레드는 지정한 수로 제한한다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Override archive path; baseline SHA-256 must still match.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--voting-methods", nargs="+", choices=VOTING, default=list(VOTING))
    parser.add_argument("--max-k", type=int, help="Evaluate every odd k from 1 through this odd maximum.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.max_k is not None and (args.max_k < 1 or args.max_k > 39999 or args.max_k % 2 == 0):
        parser.error("max-k must be an odd integer from 1 to 39,999")
    args.voting_methods = list(dict.fromkeys(args.voting_methods))
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
