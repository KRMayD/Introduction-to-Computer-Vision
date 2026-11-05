"""CV-only diagnostic: quantify voting ties without changing the original run."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from knn_cifar10.core import SCORES, ExactKNN, evaluate, load_cifar10, resolve_device, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New output directory; existing results are preserved.")
    parser.add_argument("--data", type=Path, help="Override the archive location after moving a baseline.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    source = args.results.resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    output = args.output.resolve() if args.output else source.parent / datetime.now(ZoneInfo("Asia/Seoul")).strftime("voting_analysis_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device, device_name = resolve_device(args.device, 4)
    x, y, _, _ = load_cifar10(args.data.expanduser().resolve() if args.data else Path(metadata["archive"]))
    ks = metadata["k_values"]
    near_predictions = {k: np.full(len(y), -1, dtype=np.int8) for k in ks}
    neighbors = np.empty((len(y), max(ks)), dtype=np.int8)
    tied = {k: np.zeros(len(y), dtype=bool) for k in ks}
    folds = []
    alternative_folds = []
    with np.load(source / "predictions.npz") as saved:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=metadata["seed"])
        for fold, (training, validation) in enumerate(splitter.split(x, y), 1):
            model = ExactKNN(ks, metrics=("manhattan",), device=device,
                             batch_size=metadata["batch_size"]).fit(x[training], y[training])
            labels = model.neighbor_labels(x[validation])["manhattan"]
            del model
            neighbors[validation] = labels
            votes = np.zeros((len(validation), 10), dtype=np.int32)
            rows = np.arange(len(validation))
            for position in range(max(ks)):
                votes[rows, labels[:, position]] += 1
                k = position + 1
                if k not in ks:
                    continue
                maximum = votes.max(axis=1, keepdims=True)
                winners = votes == maximum
                is_tie = winners.sum(axis=1) > 1
                original = votes.argmax(axis=1)
                np.testing.assert_array_equal(original, saved[f"oof_manhattan_k{k}"][validation])
                # Keep majority voting; for tied winning classes only, choose
                # whichever has the closest individual neighbor to the query.
                winner_positions = np.take_along_axis(winners, labels[:, :k], axis=1)
                nearest_winner = winner_positions.argmax(axis=1)
                alternative = labels[rows, nearest_winner]
                np.testing.assert_array_equal(original[~is_tie], alternative[~is_tie])
                tied[k][validation] = is_tie
                near_predictions[k][validation] = alternative
                folds.append(dict(fold=fold, k=k, n_queries=len(validation), ties=int(is_tie.sum()),
                                  original_accuracy=float((original == y[validation]).mean()),
                                  nearest_class_tiebreak_accuracy=float((alternative == y[validation]).mean())))
                alternative_folds.append(dict(fold=fold, k=k, **evaluate(y[validation], alternative)))
            print(f"Fold {fold}/5 verified against original OOF predictions; {time.perf_counter()-started:.1f}s", flush=True)
        summary = []
        for k in ks:
            original = saved[f"oof_manhattan_k{k}"]
            alternative = near_predictions[k]
            mask = tied[k]
            correct = original == y
            alt_correct = alternative == y
            summary.append(dict(k=k, n_queries=len(y), tied_queries=int(mask.sum()),
                                tie_fraction=float(mask.mean()), original_accuracy=float(correct.mean()),
                                nearest_class_tiebreak_accuracy=float(alt_correct.mean()),
                                original_accuracy_on_ties=float(correct[mask].mean()) if mask.any() else None,
                                nearest_class_accuracy_on_ties=float(alt_correct[mask].mean()) if mask.any() else None,
                                corrected_by_tiebreak=int((~correct & alt_correct).sum()),
                                harmed_by_tiebreak=int((correct & ~alt_correct).sum()),
                                changed_predictions=int((original != alternative).sum())))
        np.savez_compressed(output / "cv_voting_diagnostics.npz", y_train=y,
                            validation_fold=saved["validation_fold"], neighbor_labels=neighbors,
                            **{f"nearest_tiebreak_k{k}": pred for k, pred in near_predictions.items()})
    write_csv(output / "tie_summary.csv", summary)
    write_csv(output / "tie_folds.csv", folds)
    alternative_summary = []
    for k in ks:
        row = dict(metric="manhattan", k=k, tiebreak="nearest neighbor among tied winning classes")
        for score in SCORES:
            values = [r[score] for r in alternative_folds if r["k"] == k]
            row[score + "_mean"] = float(np.mean(values))
            row[score + "_std"] = float(np.std(values, ddof=1))
        alternative_summary.append(row)
    write_csv(output / "alternative_cv_summary.csv", alternative_summary)
    details = dict(source_results=str(source), diagnostic="Manhattan CV-only voting tie analysis",
                   original_tiebreak="smallest class ID",
                   alternative_tiebreak="closest neighbor belonging to a tied winning class",
                   original_predictions_reproduced=True, official_test_used=False,
                   n_queries=len(y), device=device, device_name=device_name,
                   elapsed_seconds=time.perf_counter()-started)
    (output / "metadata.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT", output, flush=True)
    for row in summary:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
