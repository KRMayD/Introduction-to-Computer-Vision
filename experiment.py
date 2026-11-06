"""전체 CIFAR-10에서 KNN의 5-fold 검증과 공식 테스트를 실행한다."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from knn import DEFAULT_K, METRICS, SCORES, VOTING, ExactKNN, evaluate, load_cifar10, vote, write_csv

# 기존 네 실험의 설정을 한 실행부에서 선택한다. 각 실험은 독립적으로 실행할 수 있다.
PRESETS = {
    "baseline": (METRICS, DEFAULT_K, VOTING[:1]),
    "ties": (("manhattan",), DEFAULT_K, VOTING[:2]),
    "voting": (METRICS, DEFAULT_K, VOTING),
    "k101": (METRICS, tuple(range(1, 102, 2)), ("uniform_nearest",)),
}


def resolve_device(requested, threads):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested != "auto":
            raise RuntimeError("CUDA requires PyTorch; alternatively use --device cpu.") from None
        return "cpu"
    torch.set_num_threads(threads)
    if requested == "auto" and not torch.cuda.is_available():
        return "cpu"
    device = "cuda:0" if requested == "auto" else requested
    # 컨테이너에 보이는 논리 GPU만 사용하며 CUDA_VISIBLE_DEVICES를 따른다.
    print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}", flush=True)
    return device


def tie_diagnostics(labels, truth, ks, predictions, fold):
    rows = []
    for k in ks:
        counts = np.stack([(labels[:, :k] == c).sum(axis=1) for c in range(10)], axis=1)
        tied = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
        old, new = (predictions[v][k] == truth for v in VOTING[:2])
        rows.append(dict(fold=fold, k=k, n_queries=len(truth), tied_queries=int(tied.sum()),
                         corrected_by_tiebreak=int((~old & new).sum()),
                         harmed_by_tiebreak=int((old & ~new).sum()),
                         changed_predictions=int((predictions[VOTING[0]][k] != predictions[VOTING[1]][k]).sum())))
    return rows


def run(args):
    metrics, ks, methods = PRESETS[args.experiment]
    weighted = "distance_weighted" in methods
    device = resolve_device(args.device, args.threads)
    x, y, test_x, test_y = load_cifar10(args.data)
    output = args.output or Path("runs") / args.experiment
    # 새 경로만 허용하므로 기존 결과를 덮어쓰지 않는다.
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output.resolve()}\nTrain: {len(y)}, test: {len(test_y)}, device: {device}", flush=True)
    fold_rows, diagnostics = [], []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (training, validation) in enumerate(splitter.split(x, y), 1):
        # 매 fold마다 40,000장만 참조한다. 검증 10,000장이 이웃에 섞이지 않게 한다.
        model = ExactKNN(ks, metrics, device, args.batch_size).fit(x[training], y[training])
        result = model.neighbor_labels(x[validation], return_distances=weighted)
        labels, distances = result if weighted else (result, {})
        del model
        for metric in metrics:
            # 비용이 큰 이웃 탐색은 공유하고, 투표 규칙과 k만 바꿔 공정하게 비교한다.
            predictions = {method: vote(labels[metric], ks, method, distances.get(metric)) for method in methods}
            for method in methods:
                for k in ks:
                    fold_rows.append(dict(voting=method, metric=metric, k=k, fold=fold,
                                          n_train=len(training), n_valid=len(validation),
                                          **evaluate(y[validation], predictions[method][k])))
            if args.experiment == "ties":
                diagnostics.extend(tie_diagnostics(labels[metric], y[validation], ks, predictions, fold))
        write_csv(output / "cv_folds.csv", fold_rows)
        print(f"Fold {fold}/5 complete", flush=True)

    summary = []
    for method in methods:
        for metric in metrics:
            for k in ks:
                group = [r for r in fold_rows if (r["voting"], r["metric"], r["k"]) == (method, metric, k)]
                row = dict(voting=method, metric=metric, k=k)
                for score in SCORES:
                    # CV 표에는 fold 평균과 표본 표준편차(ddof=1)를 기록한다.
                    values = [r[score] for r in group]
                    row[score + "_mean"] = float(np.mean(values))
                    row[score + "_std"] = float(np.std(values, ddof=1))
                summary.append(row)
    write_csv(output / "cv_summary.csv", summary)

    if args.experiment == "ties":
        write_csv(output / "tie_folds.csv", diagnostics)
        totals = []
        for k in ks:
            group = [r for r in diagnostics if r["k"] == k]
            total = {name: sum(r[name] for r in group) for name in group[0] if name not in ("fold", "k")}
            totals.append(dict(k=k, **total, tie_fraction=total["tied_queries"] / total["n_queries"]))
        write_csv(output / "tie_summary.csv", totals)
        return  # 동률 진단은 CV 전용이며 공식 테스트를 평가하지 않는다.

    # 테스트 정답을 보지 않고 CV 정확도로 선택한다. 동점이면 작은 k, 거리 이름 순이다.
    key = lambda row: (-row["accuracy_mean"], row["k"], row["metric"])
    best = {method: min((r for r in summary if r["voting"] == method), key=key) for method in methods}
    selected = [min((r for r in summary if (r["voting"], r["metric"]) == (method, metric)), key=key)
                for method in methods for metric in metrics]
    write_csv(output / "selected_models.csv", selected)  # 테스트 전에 선택 결과를 저장한다.
    final_ks = sorted({r["k"] for r in selected})
    # 선택이 끝난 뒤 전체 학습 50,000장을 참조해 공식 테스트 10,000장을 예측한다.
    model = ExactKNN(final_ks, metrics, device, args.batch_size).fit(x, y)
    result = model.neighbor_labels(test_x, return_distances=weighted)
    labels, distances = result if weighted else (result, {})
    test_rows = []
    for row in selected:
        method, metric, k = row["voting"], row["metric"], row["k"]
        pred = vote(labels[metric], [k], method, distances.get(metric))[k]
        test_rows.append(dict(voting=method, metric=metric, k=k,
                              selected_within_voting=best[method]["metric"] == metric,
                              n_train=len(y), n_test=len(test_y), **evaluate(test_y, pred)))
    write_csv(output / "test_results.csv", test_rows)
    for row in test_rows:
        print(f"TEST {row['voting']} {row['metric']} k={row['k']}: {row['accuracy']:.4%}", flush=True)


def main():
    # 실행 옵션을 읽은 뒤 선택한 실험을 수행한다. 기본값은 세 투표 방식 비교다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=PRESETS, default="voting")
    parser.add_argument("--data", type=Path, required=True, help="CIFAR-10 Python .tar or .tar.gz")
    parser.add_argument("--output", type=Path, help="New output directory (default: runs/<experiment>)")
    parser.add_argument("--device", default="cpu", help="cpu, auto, or visible GPU such as cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    # 공유 서버의 CPU를 과도하게 쓰지 않도록 BLAS 스레드 수를 제한한다.
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
