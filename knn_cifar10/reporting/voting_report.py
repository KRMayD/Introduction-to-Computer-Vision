"""Create tables, graphs and a Korean report from a completed weighted run."""

import argparse
import csv
import html
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from knn_cifar10.core import CLASS_NAMES, METRICS, SCORES


NAMES = {"uniform_class_id": "Uniform / class-ID tie", "uniform_nearest": "Uniform / nearest tie",
         "distance_weighted": "Inverse-distance"}
COLORS = {"uniform_class_id": "#94a3b8", "uniform_nearest": "#059669", "distance_weighted": "#2563eb"}


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    if (root / "report.html").exists():
        raise FileExistsError("Report already exists; existing experiment outputs are preserved.")
    output = root / "figures"
    output.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    summary = read_csv(root / "cv_summary.csv")
    weighted = [r for r in summary if r["voting"] == "distance_weighted"]
    test = read_csv(root / "weighted_test_results.csv")
    comparison = read_csv(root / "voting_comparison.csv")
    all_test = read_csv(root / "test_results.csv")
    metadata = json.loads((root / "metadata.json").read_text())
    selected = metadata["selected_by_voting"]["distance_weighted"]
    images = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(path) as image:
            image.verify()
        images.append(name)

    def table(name, title, subtitle, headers, rows, widths=None, highlight=(), note=""):
        height = max(3.7, 1.8 + .38 * (len(rows) + 1))
        fig = plt.figure(figsize=(15, height), facecolor="white")
        fig.text(.04, 1-.22/height, title, fontsize=19, weight="bold", va="top", color="#14213d")
        fig.text(.04, 1-.66/height, subtitle, fontsize=10.5, va="top", color="#475569")
        ax = fig.add_axes([.035, .62/height, .93, (height-1.72)/height])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="center", bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(10.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("white")
            if r == 0:
                cell.set_facecolor("#14213d")
                cell.set_text_props(color="white", weight="bold", fontsize=10)
            else:
                cell.set_facecolor("#eef3f9" if r % 2 else "white")
                if r-1 in highlight:
                    cell.set_facecolor("#dcecff")
                    cell.set_text_props(weight="bold")
        fig.text(.04, .25/height, note, fontsize=9, color="#475569", va="bottom")
        save(fig, name)

    headers = ["Distance", "k", "Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1 (%)"]
    table("01_weighted_test_results.png", "Inverse-distance KNN | Official test results",
          "50,000 training + 10,000 test images | Raw 3,072-dimensional pixels | k selected by 5-fold CV",
          headers, [[r["metric"].title(), r["k"], *[f"{float(r[s])*100:.2f}" for s in SCORES]] for r in test],
          highlight=[i for i, r in enumerate(test) if r["selected_within_voting"] == "True"],
          note="Blue row: distance-weighted model selected using CV. Inverse-distance weights; only zero-distance neighbors vote when present.")
    table("02_weighted_cv_results.png", "Inverse-distance KNN | All 24 CV settings",
          "Stratified 5-fold CV | 40,000 training + 10,000 validation images per fold | Same splits as baseline",
          headers, [[r["metric"].title(), r["k"], *[f"{float(r[s+'_mean'])*100:.2f} ± {float(r[s+'_std'])*100:.2f}" for s in SCORES]] for r in weighted],
          highlight=[i for i, r in enumerate(weighted) if r["metric"] == selected["metric"] and int(r["k"]) == selected["k"]],
          note="Values: mean ± sample standard deviation (ddof=1), in %. Blue row: best mean CV accuracy.")
    table("03_voting_comparison.png", "CIFAR-10 | Voting rules compared",
          "Each voting rule selects its distance and k using identical 5-fold CV; all models use the full dataset",
          ["Voting", "Distance", "k", "CV Accuracy\nmean ± SD (%)", "Test\nAccuracy (%)", "Macro\nPrecision (%)", "Macro\nRecall (%)", "Macro\nF1 (%)"],
          [[NAMES[r["voting"]], r["metric"].title(), r["k"], f"{float(r['cv_accuracy'])*100:.2f} ± {float(r['cv_accuracy_std'])*100:.2f}",
            *[f"{float(r[s])*100:.2f}" for s in SCORES]] for r in comparison],
          widths=[.25, .12, .05, .18, .10, .10, .10, .10], highlight=[2],
          note="Test set: 10,000 images; training set: 50,000. The original baseline predictions were reproduced exactly.")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        for vote, label in NAMES.items():
            rows = [r for r in summary if r["voting"] == vote and r["metric"] == metric]
            ks = [int(r["k"]) for r in rows]
            ax.errorbar(ks, [float(r["accuracy_mean"])*100 for r in rows],
                        yerr=[float(r["accuracy_std"])*100 for r in rows], marker="o", capsize=2,
                        label=label, color=COLORS[vote], linewidth=1.8)
        ax.set(title=metric.title(), xlabel="k", ylabel="CV accuracy (%)")
        ax.set_xticks(ks)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Same full-data 5-fold CV | Voting rules, distances and k", fontsize=17)
    save(fig, "04_cv_voting_curves.png")
    cm_rows = read_csv(root / "confusion_distance_weighted.csv")
    matrix = np.array([[int(r[c]) for c in CLASS_NAMES] for r in cm_rows])
    fig, ax = plt.subplots(figsize=(11, 9.5), constrained_layout=True)
    pic = ax.imshow(matrix, cmap="Blues", vmin=0)
    for r in range(10):
        for c in range(10):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=10,
                    color="white" if matrix[r, c] > matrix.max()*.5 else "#14213d")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"Inverse-distance | {selected['metric']}, k={selected['k']} | Official test")
    fig.colorbar(pic, ax=ax, label="Images", shrink=.85)
    save(fig, "05_weighted_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    positions = np.arange(4)
    for i, r in enumerate(comparison):
        bars = ax.bar(positions+(i-1)*.25, [float(r[s])*100 for s in SCORES], width=.24,
                      color=COLORS[r["voting"]], label=NAMES[r["voting"]])
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xticks(positions, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
    ax.set(ylabel="Score (%)", ylim=(0, 58), title="Official test scores | Each model selected using CV")
    ax.legend(ncols=3, fontsize=9)
    ax.yaxis.grid(alpha=.2)
    ax.set_axisbelow(True)
    save(fig, "06_test_voting_metrics.png")

    def html_table(rows, columns, formatter=None):
        def value(row, key):
            return html.escape(formatter(row, key) if formatter else str(row[key]))
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>" + "".join(
            "<tr>"+"".join(f"<td>{value(r,c)}</td>" for c in columns)+"</tr>" for r in rows) + "</tbody></table>"

    def percent_format(row, key):
        if key in SCORES or key in ("cv_accuracy", "cv_accuracy_std", "precision", "recall", "f1-score"):
            return f"{float(row[key])*100:.2f}%"
        return NAMES.get(row[key], str(row[key]))

    cv_display = [dict(metric=r["metric"], k=r["k"], **{s:f"{float(r[s+'_mean'])*100:.2f} ± {float(r[s+'_std'])*100:.2f}%" for s in SCORES}) for r in weighted]
    weighted_selected_test = next(r for r in comparison if r["voting"] == "distance_weighted")
    original_selected_test = next(r for r in comparison if r["voting"] == "uniform_class_id")
    delta = (float(weighted_selected_test["accuracy"])-float(original_selected_test["accuracy"]))*100
    class_rows = [r for r in read_csv(root/"classification_report.csv") if r["voting"] == "distance_weighted"]
    report = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIFAR-10 거리 가중 KNN 전체 데이터 실험</title><style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;line-height:1.7;color:#14213d}}table{{border-collapse:collapse;font-size:14px;margin:20px 0}}th,td{{border:1px solid #d9e1ec;padding:8px 10px;text-align:right}}th{{background:#edf2f8}}td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#edf2f8}}.result{{background:#e8f1ff;padding:16px}}</style>
<h1>CIFAR-10 거리 가중 KNN 전체 데이터 실험</h1>
<p class="result">거리 가중 투표의 CV 최적 설정: <strong>{selected['metric']}, k={selected['k']}</strong>.<br>
CV Accuracy {selected['accuracy_mean']:.2%} ± {selected['accuracy_std']*100:.2f}%p.
공식 테스트 Accuracy <strong>{float(weighted_selected_test['accuracy']):.2%}</strong>.
기존 일반 투표 최적 모델 대비 테스트 정확도 차이: <strong>{delta:+.2f}%p</strong>.</p>
<h2>실험 설정과 검증</h2><p>학습 50,000장 전체, 공식 테스트 10,000장 전체, 원본 픽셀 3,072차원.
StratifiedKFold(n_splits=5, shuffle=True, random_state={metadata['seed']})로 매 fold 40,000장 학습 및 10,000장 검증을 수행했다.
모든 학습 이미지는 정확히 한 번 검증에 사용됐다. 거리 3개 × k 8개로 가중 투표 120개 fold 결과를 계산했다.
세 투표 방식의 비교까지 합하면 총 360개 fold 결과이다. k={metadata['k_values']}.</p>
<p>입력 데이터 SHA-256, fold 분할 및 기존 일반 투표의 OOF·테스트 예측이 원본 실험과 동일함을 확인했다.
거리별 및 투표 방식별 설정은 평균 CV 정확도로 선택하여 테스트 예측 전에 metadata.json에 저장했다.
각 모델의 최종 테스트 평가는 학습 50,000장을 모두 사용했다.</p>
<h2>가중 투표 구현</h2><p>각 이웃의 가중치는 w=1/d이며 클래스별 가중치 합이 가장 큰 클래스를 선택한다.
Euclidean은 실제 거리의 역수를 사용하며 제곱 거리의 역수를 사용하지 않는다.
거리 0인 이웃이 있으면 해당 이웃들만 같은 가중치로 투표한다.
가중치 합이 동률이면(rtol=1e-12, atol=0) 동률 후보 중 가장 가까운 이웃의 클래스를 선택하고, 거리까지 같으면 안정 정렬된 학습 인덱스 순서를 따른다.
일반 투표는 작은 클래스 ID 동률 처리와 가까운 이웃 동률 처리의 두 방식을 비교했다.</p>
<h2>투표 방식별 CV 최적 모델 비교</h2>
{html_table(comparison,['voting','metric','k','cv_accuracy','cv_accuracy_std',*SCORES],percent_format)}
<img src="figures/03_voting_comparison.png"><img src="figures/06_test_voting_metrics.png">
<h2>거리 가중 투표: 전체 24개 설정의 5-fold 결과</h2><p>평균 ± 표본 표준편차(ddof=1)이며 신뢰구간이 아니다.
Precision·Recall·F1은 클래스별 지표를 동일 가중 평균한 macro 값이다.</p>
{html_table(cv_display,['metric','k',*SCORES])}<img src="figures/04_cv_voting_curves.png">
<h2>거리 가중 투표: 거리별 CV 최적 k의 테스트 결과</h2>
{html_table(test,['metric','k','selected_within_voting',*SCORES],percent_format)}
<h2>모든 투표 방식과 거리의 테스트 결과</h2>
{html_table(all_test,['voting','metric','k','selected_within_voting',*SCORES],percent_format)}
<h2>가중 투표 최적 모델의 클래스별 결과</h2>
{html_table(class_rows,['class_name','precision','recall','f1-score','support'],percent_format)}
<img src="figures/05_weighted_confusion_matrix.png">
<h2>산출물과 재현</h2><p>실험 실행 시간: {metadata['elapsed_seconds']:.1f}초. 단일 {metadata['device_name']} GPU 사용.</p>
<p><code>{html.escape(metadata['command'])}</code></p>
<p><a href="weighted_cv_summary.csv">가중 투표 CV 요약</a> · <a href="cv_folds.csv">전체 fold 지표</a> ·
<a href="weighted_test_results.csv">가중 투표 테스트 결과</a> · <a href="voting_comparison.csv">투표 방식 비교</a> ·
<a href="metadata.json">실험 메타데이터</a> · <a href="weighted_results_bundle.zip">보고서·이미지·CSV 묶음</a></p></html>"""
    (root / "report.html").write_text(report, encoding="utf-8")
    with zipfile.ZipFile(root/"weighted_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [root/"report.html", root/"metadata.json", *root.glob("*.csv"), *output.glob("*.png")]:
            bundle.write(path, str(path.relative_to(root)))
    with zipfile.ZipFile(root/"weighted_results_bundle.zip") as bundle:
        assert bundle.testzip() is None
    print("REPORT",root/"report.html")
    print("IMAGES",len(images),"PNGs, 300dpi")
    print("COMPARISON",json.dumps(comparison,ensure_ascii=False))


if __name__ == "__main__":
    main()
