"""Full CIFAR-10 5-fold experiments with configurable KNN voting rules and k."""

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
    predictions = {}
    if "uniform_class_id" in voting_methods:
        predictions["uniform_class_id"] = majority_predictions(labels, ks)
    if "uniform_nearest" in voting_methods:
        predictions["uniform_nearest"] = nearest_tie_predictions(labels, ks)
    if "distance_weighted" in voting_methods:
        predictions["distance_weighted"] = distance_weighted_predictions(labels, distances, ks)
    return predictions


def run(args):
    started = time.perf_counter()
    source = args.baseline.resolve()
    baseline_meta = json.loads((source / "metadata.json").read_text())
    voting_methods = tuple(args.voting_methods)
    use_distances = "distance_weighted" in voting_methods
    prefix = f"nearest_k{args.max_k or max(baseline_meta['k_values'])}" if voting_methods == ("uniform_nearest",) else "weighted_full"
    stamp = prefix + datetime.now(ZoneInfo("Asia/Seoul")).strftime("_%Y%m%d_%H%M%S")
    output = args.output.resolve() if args.output else source.parent / stamp
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
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != baseline_meta["archive_sha256"]:
        raise ValueError("Dataset hash differs from the baseline experiment.")
    ks = list(range(1, args.max_k + 1, 2)) if args.max_k else baseline_meta["k_values"]
    shared_ks = [k for k in baseline_meta["k_values"] if k in ks]
    seed = baseline_meta["seed"]
    batch_size = baseline_meta["batch_size"]
    with np.load(source / "predictions.npz") as original:
        np.testing.assert_array_equal(original["y_train"], y)
        np.testing.assert_array_equal(original["y_test"], test_y)
        baseline_folds = original["validation_fold"].copy()
        baseline_oof = {(metric, k): original[f"oof_{metric}_k{k}"].copy() for metric in METRICS for k in shared_ks}
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
    oof = {(vote, metric, k): np.full(len(y), -1, dtype=np.int8) for vote in voting_methods for metric in METRICS for k in ks}
    neighbor_labels = {metric: np.empty((len(y), max(ks)), dtype=np.int8) for metric in METRICS}
    neighbor_distances = {metric: np.empty((len(y), max(ks)), dtype=np.float64) for metric in METRICS} if use_distances else {}
    assignment = np.zeros(len(y), dtype=np.int8)
    fold_rows = []
    for fold, (training, validation) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y), 1):
        np.testing.assert_array_equal(np.flatnonzero(baseline_folds == fold), validation)
        assignment[validation] = fold
        log(f"Fold {fold}/5: train={len(training)}, validation={len(validation)}")
        classifier = ExactKNN(ks, device=device, batch_size=batch_size).fit(x[training], y[training])
        result = classifier.neighbor_labels(x[validation], lambda n, total: log(f"Fold {fold}: {n}/{total} queries"), return_distances=use_distances)
        labels, distances = result if use_distances else (result, {})
        del classifier
        for metric in METRICS:
            neighbor_labels[metric][validation] = labels[metric]
            if use_distances:
                neighbor_distances[metric][validation] = distances[metric]
            if shared_ks:
                original_predictions = majority_predictions(labels[metric], shared_ks)
                for k in shared_ks:
                    np.testing.assert_array_equal(original_predictions[k], baseline_oof[metric, k][validation])
            predictions = all_predictions(labels[metric], distances.get(metric), ks, voting_methods)
            for vote in voting_methods:
                for k in ks:
                    pred = predictions[vote][k]
                    if vote == "uniform_class_id" and k in shared_ks:
                        np.testing.assert_array_equal(pred, baseline_oof[metric, k][validation])
                    oof[vote, metric, k][validation] = pred
                    fold_rows.append(dict(voting=vote, metric=metric, k=k, fold=fold,
                                          n_train=len(training), n_valid=len(validation), **evaluate(y[validation], pred)))
        write_csv(output / "cv_folds.csv", fold_rows)
        log(f"Fold {fold}/5 complete; original uniform predictions reproduced exactly")
    assert np.all(assignment > 0) and all(np.all(pred >= 0) for pred in oof.values())
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
                row.update({score+"_oof": value for score, value in evaluate(y, oof[vote, metric, k]).items()})
                summary.append(row)
    write_csv(output / "cv_summary.csv", summary)
    if use_distances:
        write_csv(output / "weighted_cv_summary.csv", [r for r in summary if r["voting"] == "distance_weighted"])
    write_csv(output / "fold_assignments.csv", [dict(train_index=i, label=int(y[i]), validation_fold=int(assignment[i])) for i in range(len(y))])
    np.savez_compressed(output / "oof_neighbors.npz", validation_fold=assignment, y_train=y,
                        **{f"{m}_labels": a for m, a in neighbor_labels.items()},
                        **{f"{m}_distances": a for m, a in neighbor_distances.items()})
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
    final_ks = sorted({r["k"] for r in per_metric.values()})
    classifier = ExactKNN(final_ks, device=device, batch_size=batch_size).fit(x, y)
    result = classifier.neighbor_labels(test_x, lambda n, total: log(f"Full-train test: {n}/{total} queries"), return_distances=use_distances)
    test_labels, test_distances = result if use_distances else (result, {})
    del classifier
    test_predictions, test_rows = {}, []
    for metric in METRICS:
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
    class_rows = []
    for vote in voting_methods:
        metric, k = selected[vote]["metric"], selected[vote]["k"]
        pred = test_predictions[vote, metric, k]
        report = classification_report(test_y, pred, labels=np.arange(10), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        class_rows.extend(dict(voting=vote, metric=metric, k=k, class_name=name, **report[name]) for name in CLASS_NAMES)
        matrix = confusion_matrix(test_y, pred, labels=np.arange(10))
        write_csv(output / f"confusion_{vote}.csv", [dict(actual=name, **dict(zip(CLASS_NAMES, map(int, matrix[i])))) for i, name in enumerate(CLASS_NAMES)])
    write_csv(output / "classification_report.csv", class_rows)
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
