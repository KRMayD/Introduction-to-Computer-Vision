"""Export saved CIFAR-10 KNN results as publication-size PNGs; no model rerun."""

import argparse
import csv
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


METRICS = ("euclidean", "manhattan", "cosine")
COLORS = {"euclidean": "#2563eb", "manhattan": "#dc2626", "cosine": "#059669"}
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
LABELS = ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source, analysis = args.results.resolve(), args.analysis.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("figures_%Y%m%d_%H%M%S")
    output = args.output.resolve() if args.output else source.parent.parent / "exports" / stamp
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / ".mplconfig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#14213d",
                         "axes.titleweight": "bold", "axes.titlesize": 13,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    manifest = []

    def save(fig, filename, description):
        path = output / filename
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white",
                    metadata={"Description": description})
        plt.close(fig)
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            manifest.append(dict(filename=filename, description=description,
                                 width=picture.width, height=picture.height,
                                 dpi=list(picture.info.get("dpi", ()))))
        print(filename, flush=True)

    def table(filename, title, subtitle, headers, rows, description,
              highlight=(), widths=None, footnote="", width=13.5):
        height = max(3.7, 1.8 + 0.38 * (len(rows) + 1))
        fig = plt.figure(figsize=(width, height))
        fig.text(0.04, 1 - 0.22 / height, title, fontsize=19, weight="bold", va="top")
        fig.text(0.04, 1 - 0.66 / height, subtitle, fontsize=10.5, va="top", color="#475569")
        ax = fig.add_axes([0.035, 0.62 / height, 0.93, (height - 1.72) / height])
        ax.axis("off")
        artist = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                          cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        artist.auto_set_font_size(False)
        artist.set_fontsize(10.5)
        for (r, c), cell in artist.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(1.3)
            if r == 0:
                cell.set_facecolor("#14213d")
                cell.set_text_props(color="white", weight="bold", fontsize=10)
            else:
                cell.set_facecolor("#eef3f9" if r % 2 else "#ffffff")
                if r - 1 in highlight:
                    cell.set_facecolor("#dcecff")
                    cell.set_text_props(weight="bold")
        fig.text(0.04, 0.26 / height, footnote, fontsize=9, color="#475569", va="bottom")
        save(fig, filename, description)

    cv = read_csv(source / "cv_summary.csv")
    folds = read_csv(source / "cv_folds.csv")
    test = read_csv(source / "test_results.csv")
    per_class = read_csv(source / "classification_report.csv")
    cm_rows = read_csv(source / "confusion_matrix.csv")
    ties = read_csv(analysis / "tie_summary.csv")
    metadata = json.loads((source / "metadata.json").read_text())
    assert len(cv) == 24 and len(folds) == 120 and len(test) == 3 and len(per_class) == 10
    assert metadata["train_images"] == 50000 and metadata["test_images"] == 10000
    selected = metadata["selected_model"]
    score_headers = ["Accuracy\n(%)", "Macro Precision\n(%)", "Macro Recall\n(%)", "Macro F1\n(%)"]

    table("01_test_results_table.png", "CIFAR-10 | Official test results",
          "Full training set: 50,000 images  |  Official test set: 10,000 images  |  Raw pixels: 3,072 features",
          ["Distance", "k", *score_headers],
          [[r["metric"].title(), r["k"], *[f"{float(r[s])*100:.2f}" for s in SCORES]] for r in test],
          "공식 테스트 결과표: 거리별 CV 최적 k", highlight=[i for i, r in enumerate(test) if r["selected_by_cv"] == "True"],
          widths=[.18, .06, .18, .20, .18, .20],
          footnote="Blue row: model selected by mean CV accuracy before test evaluation. Original voting rule: smallest class ID on a tie.")

    table("02_cv_results_table.png", "CIFAR-10 | All 24 cross-validation settings",
          "Stratified 5-fold CV  |  Each fold: 40,000 training + 10,000 validation images  |  Seed: 42",
          ["Distance", "k", *score_headers],
          [[r["metric"].title(), r["k"], *[f"{float(r[s+'_mean'])*100:.2f} ± {float(r[s+'_std'])*100:.2f}" for s in SCORES]] for r in cv],
          "전체 24개 설정의 교차검증 평균 및 표준편차", highlight=[i for i, r in enumerate(cv) if r["metric"] == selected["metric"] and int(r["k"]) == selected["k"]],
          widths=[.18, .06, .18, .20, .18, .20],
          footnote="Values: mean ± sample standard deviation (ddof=1), in %. Standard deviations are not confidence intervals.")

    table("03_classification_report_table.png", "CIFAR-10 | Per-class test performance",
          "CV-selected model: Manhattan, k=1  |  Original voting rule  |  1,000 official test images per class",
          ["Class", "Precision (%)", "Recall (%)", "F1 (%)", "Support"],
          [[r["class_name"], *[f"{float(r[s])*100:.2f}" for s in ("precision", "recall", "f1-score")], f"{int(float(r['support'])):,}"] for r in per_class],
          "최종 선택 모델의 클래스별 테스트 성능", widths=[.24, .20, .20, .20, .16],
          footnote="Class-level precision, recall and F1. Support is the number of true test images.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    for ax, score, label in zip(axes.flat, SCORES, LABELS):
        for metric in METRICS:
            rows = [r for r in cv if r["metric"] == metric]
            ks = [int(r["k"]) for r in rows]
            ax.errorbar(ks, [float(r[score+"_mean"])*100 for r in rows],
                        yerr=[float(r[score+"_std"])*100 for r in rows], marker="o", capsize=3,
                        color=COLORS[metric], label=metric.title(), linewidth=1.8)
        ax.set(xlabel="k", ylabel=f"{label} (%)", title=label)
        ax.set_xticks(ks)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Original voting rule | 5-fold CV mean ± standard deviation", fontsize=17)
    save(fig, "04_cv_curves.png", "거리와 k에 따른 4개 CV 지표 그래프: 기존 동률 규칙")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    positions = np.arange(4)
    for i, row in enumerate(test):
        values = [float(row[s])*100 for s in SCORES]
        bars = ax.bar(positions + (i-1)*.25, values, width=.24, color=COLORS[row["metric"]],
                      label=f"{row['metric'].title()} (k={row['k']})")
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xticks(positions, LABELS)
    ax.set(ylabel="Score (%)", ylim=(0, 55), title="Official test set | 10,000 images, full 50,000-image training set")
    ax.legend(loc="upper right", ncols=3, fontsize=10)
    ax.yaxis.grid(alpha=.2)
    ax.set_axisbelow(True)
    save(fig, "05_test_metrics.png", "거리별 공식 테스트 지표 비교 그래프")

    matrix = np.array([[int(r[c]) for c in CLASSES] for r in cm_rows])
    assert matrix.sum() == 10000 and np.all(matrix.sum(1) == 1000)
    selected_test = next(r for r in test if r["selected_by_cv"] == "True")
    assert matrix.trace() == round(float(selected_test["accuracy"])*10000)
    fig, ax = plt.subplots(figsize=(11, 9.5), constrained_layout=True)
    pic = ax.imshow(matrix, cmap="Blues", vmin=0)
    for r in range(10):
        for c in range(10):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[r, c] > matrix.max()*.5 else "#14213d")
    ax.set_xticks(range(10), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Manhattan, k=1 | Official test confusion matrix")
    fig.colorbar(pic, ax=ax, label="Images", shrink=.85)
    save(fig, "06_confusion_matrix.png", "Manhattan k=1의 공식 테스트 혼동행렬")

    counts = matrix.sum(0)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    bars = ax.bar(CLASSES, counts, color=["#2563eb" if n >= 1000 else "#64748b" for n in counts], width=.7)
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.axhline(1000, color="#dc2626", linestyle="--", label="True count: 1,000 per class")
    ax.set(ylabel="Images", ylim=(0, 1950), title="Manhattan, k=1 | Predicted class distribution on the official test set")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.yaxis.grid(alpha=.2)
    ax.set_axisbelow(True)
    save(fig, "07_prediction_distribution.png", "실제 균등 분포와 모델의 클래스별 예측 수 비교")

    table("08_voting_tie_table.png", "Manhattan | Voting tie diagnostic",
          "Same 50,000 OOF queries, folds and nearest neighbors; only the vote tie rule changes",
          ["k", "Voting ties\n(%)", "Original CV\naccuracy (%)", "Nearest-tie CV\naccuracy (%)", "Change\n(%p)"],
          [[r["k"], f"{float(r['tie_fraction'])*100:.2f}", f"{float(r['original_accuracy'])*100:.2f}",
            f"{float(r['nearest_class_tiebreak_accuracy'])*100:.2f}",
            f"+{(float(r['nearest_class_tiebreak_accuracy'])-float(r['original_accuracy']))*100:.2f}"] for r in ties],
          "동률 발생률과 동률 규칙 변경에 따른 CV 정확도 비교", highlight=[2],
          footnote="Original: smallest class ID. Alternative: closest neighbor in a tied winning class. CV diagnostic only; no alternative test score.")

    ks = [int(r["k"]) for r in ties]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].plot(ks, [float(r["original_accuracy"])*100 for r in ties], "o-", color="#b45309", label="Tie: smallest class ID")
    axes[0].plot(ks, [float(r["nearest_class_tiebreak_accuracy"])*100 for r in ties], "o-", color="#2563eb", label="Tie: closest neighbor in winning classes")
    axes[0].set(xlabel="k", ylabel="Mean CV accuracy (%)", title="Same neighbors; different voting tie rule")
    axes[0].legend(fontsize=8.5)
    axes[1].plot(ks, [float(r["tie_fraction"])*100 for r in ties], "o-", color="#7c3aed")
    axes[1].set(xlabel="k", ylabel="Queries with a voting tie (%)", title="Vote ties on all 50,000 OOF queries")
    for ax in axes:
        ax.set_xticks(ks)
        ax.grid(alpha=.2)
    fig.suptitle("Manhattan | Additional CV diagnostic; alternative rule not test-evaluated", fontsize=15)
    save(fig, "09_voting_tie_graph.png", "추가 동률 진단: 기존/대안 CV 곡선과 투표 동률 발생률")

    for number, metric in enumerate(METRICS, 10):
        rows = [r for r in folds if r["metric"] == metric]
        rows.sort(key=lambda r: (int(r["fold"]), int(r["k"])))
        assert len(rows) == 40
        table(f"{number:02d}_{metric}_fold_results.png", f"{metric.title()} | All 40 fold-level results",
              "5 folds × 8 k settings  |  Each fold: 40,000 training + 10,000 validation images",
              ["Fold", "k", *score_headers],
              [[r["fold"], r["k"], *[f"{float(r[s])*100:.2f}" for s in SCORES]] for r in rows],
              f"{metric} 거리의 전체 fold별 지표 40개",
              footnote="Original voting rule. Precision, recall and F1 use macro averaging over all 10 classes.")

    table("13_external_k_comparison.png", "Does accuracy always decrease as k increases?",
          "Published settings differ; these are qualitative comparisons, not an accuracy ranking",
          ["Source", "Setting", "Best reported k", "Observed result"],
          [["Our original CV", "50,000 raw-pixel images; L1", "1", "Best among our 8 tested k values"],
           ["Our tie diagnostic", "Same L1 neighbors; nearest tie rule", "5", "CV accuracy improves over k=1"],
           ["Gallego et al. (2018)", "CIFAR-10 raw data; no label noise", "1", "Table 3 reports k=1 as best"],
           ["J. Abbott (2019)", "5,000 train / 500 test; L2", "10", "k=1/5/10: 27.4/27.8/28.2% test acc."],
           ["Stanford CS231n", "Illustrative 5-fold CV example", "About 7", "Intermediate k is best in the example"]],
          "외부 실험의 k 변화 비교: 조건별로 최적 k가 다름", widths=[.20, .30, .13, .37], width=17,
          footnote="Sources: mdpi.com/2076-3417/8/11/2086 (Table 3); abbottwhitley.github.io (2019-09-19 KNN); cs231n.github.io/classification/.")

    source_links = {
        "Gallego et al. 2018": "https://www.mdpi.com/2076-3417/8/11/2086",
        "Julian Abbott 2019": "https://abbottwhitley.github.io/documentation/2019/09/19/K-Nearest-Neighbor-Classifier.html",
        "Stanford CS231n": "https://cs231n.github.io/classification/",
    }
    details = dict(dpi=300, original_results=str(source), voting_analysis=str(analysis),
                   images=manifest, external_sources=source_links,
                   notes=["All original training and test images were used.",
                          "Alternative voting-rule results are CV-only.",
                          "External studies use different evaluation settings."])
    (output / "manifest.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["CIFAR-10 KNN 결과 이미지", "", "모든 이미지는 300dpi PNG입니다.",
             "01~07, 10~12: 기존 전체 데이터 실험 결과", "08~09: 추가 동률 규칙 진단(CV 전용)",
             "13: 외부 실험의 최적 k 비교. 데이터 규모와 평가 조건이 서로 다릅니다.",
             "", "파일 목록"]
    lines += [f"{r['filename']}: {r['description']} ({r['width']}×{r['height']})" for r in manifest]
    lines += ["", "원본 실험: " + str(source), "추가 진단: " + str(analysis), "", "외부 출처"]
    lines += [f"{name}: {url}" for name, url in source_links.items()]
    (output / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = output / "cifar10_knn_images.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for record in manifest:
            bundle.write(output / record["filename"], record["filename"])
        for name in ("README.txt", "manifest.json"):
            bundle.write(output / name, name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len([name for name in bundle.namelist() if name.endswith(".png")]) == 13
    print(f"OUTPUT {output}", flush=True)
    print(f"VERIFIED {len(manifest)} PNGs; ZIP {archive.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
