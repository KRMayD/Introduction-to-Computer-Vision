"""CV-only diagnostic: quantify voting ties without changing the original run."""

# 같은 Manhattan 이웃 목록에서 동률 해결 규칙만 바꾸어 영향과 빈도를 측정한다.
# 이 분석은 학습 데이터의 5-fold 검증 전용이며 공식 테스트 점수를 계산하지 않는다.
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
    # 공통 로더가 테스트 데이터도 읽지만 여기서는 반환값을 버리고 학습 데이터만 분석한다.
    x, y, _, _ = load_cifar10(args.data.expanduser().resolve() if args.data else Path(metadata["archive"]))
    ks = metadata["k_values"]
    # 각 이미지의 OOF 예측과 동률 여부를 원래 학습 인덱스 위치에 저장한다.
    near_predictions = {k: np.full(len(y), -1, dtype=np.int8) for k in ks}
    neighbors = np.empty((len(y), max(ks)), dtype=np.int8)
    tied = {k: np.zeros(len(y), dtype=bool) for k in ks}
    folds = []
    alternative_folds = []
    with np.load(source / "predictions.npz") as saved:
        # 기본 실험과 동일한 seed로 fold를 재현한다.
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=metadata["seed"])
        for fold, (training, validation) in enumerate(splitter.split(x, y), 1):
            model = ExactKNN(ks, metrics=("manhattan",), device=device,
                             batch_size=metadata["batch_size"]).fit(x[training], y[training])
            labels = model.neighbor_labels(x[validation])["manhattan"]
            del model
            neighbors[validation] = labels
            # 한 이웃당 1표씩 누적하며, 요청한 k에 도달했을 때만 결과를 기록한다.
            votes = np.zeros((len(validation), 10), dtype=np.int32)
            rows = np.arange(len(validation))
            for position in range(max(ks)):
                votes[rows, labels[:, position]] += 1
                k = position + 1
                if k not in ks:
                    continue
                maximum = votes.max(axis=1, keepdims=True)
                winners = votes == maximum
                # 최대 표를 받은 클래스가 두 개 이상이면 투표 동률이다.
                is_tie = winners.sum(axis=1) > 1
                original = votes.argmax(axis=1)
                # 먼저 기존의 작은 클래스 번호 선택 예측이 정확히 재현되는지 확인한다.
                np.testing.assert_array_equal(original, saved[f"oof_manhattan_k{k}"][validation])
                # Keep majority voting; for tied winning classes only, choose
                # whichever has the closest individual neighbor to the query.
                # 동률 후보에 속한 이웃만 남긴 뒤 거리순으로 첫 후보를 선택한다.
                winner_positions = np.take_along_axis(winners, labels[:, :k], axis=1)
                nearest_winner = winner_positions.argmax(axis=1)
                alternative = labels[rows, nearest_winner]
                # 동률이 없었던 이미지의 예측은 규칙 변경 전후에 동일해야 한다.
                np.testing.assert_array_equal(original[~is_tie], alternative[~is_tie])
                tied[k][validation] = is_tie
                near_predictions[k][validation] = alternative
                folds.append(dict(fold=fold, k=k, n_queries=len(validation), ties=int(is_tie.sum()),
                                  original_accuracy=float((original == y[validation]).mean()),
                                  nearest_class_tiebreak_accuracy=float((alternative == y[validation]).mean())))
                alternative_folds.append(dict(fold=fold, k=k, **evaluate(y[validation], alternative)))
            print(f"Fold {fold}/5 verified against original OOF predictions; {time.perf_counter()-started:.1f}s", flush=True)
        # 모든 fold를 합쳐 동률 비율과 정확도 변화를 구한다.
        summary = []
        for k in ks:
            original = saved[f"oof_manhattan_k{k}"]
            alternative = near_predictions[k]
            mask = tied[k]
            correct = original == y
            alt_correct = alternative == y
            # corrected: 오답→정답, harmed: 정답→오답. 두 수의 차이가 전체 정답 수 변화다.
            # 동률이 전혀 없는 k=1에서는 동률 부분집합의 정확도를 None으로 기록한다.
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
    # 대안 규칙의 4개 지표도 같은 5개 fold에서 평균과 표본 표준편차로 요약한다.
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
