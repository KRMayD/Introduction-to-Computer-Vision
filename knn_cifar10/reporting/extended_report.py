"""Export the uniform-vote, nearest-tie KNN search through k=101."""

import argparse
import csv
import html
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from knn_cifar10.core import CLASS_NAMES, METRICS, SCORES, write_csv


def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    root, previous = args.results.resolve(), args.previous.resolve()
    if (root / "report.html").exists():
        raise FileExistsError("A report already exists; existing outputs are preserved.")
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["voting_methods"] == ["uniform_nearest"]
    summary, tests = read(root/"cv_summary.csv"), read(root/"test_results.csv")
    classes = read(root/"classification_report.csv")
    ks = metadata["k_values"]
    assert len(summary) == len(ks)*3
    best = metadata["selected_by_voting"]["uniform_nearest"]
    index = {(r["metric"], int(r["k"])): r for r in summary}
    old_summary = [r for r in read(previous/"cv_summary.csv") if r["voting"] == "uniform_nearest"]
    old_test = {r["metric"]: r for r in read(previous/"test_results.csv") if r["voting"] == "uniform_nearest"}
    comparisons = []
    for r in tests:
        old = min([s for s in old_summary if s["metric"] == r["metric"]], key=lambda s:(-float(s["accuracy_mean"]), int(s["k"])))
        new = index[r["metric"], int(r["k"])]
        comparisons.append(dict(metric=r["metric"], previous_k=int(old["k"]), previous_cv_accuracy=float(old["accuracy_mean"]),
                                previous_test_accuracy=float(old_test[r["metric"]]["accuracy"]), new_k=int(r["k"]),
                                new_cv_accuracy=float(new["accuracy_mean"]), new_test_accuracy=float(r["accuracy"])))
    write_csv(root/"range_comparison.csv", comparisons)
    figures = root/"figures"
    figures.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(figures/".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.spines.top":False, "axes.spines.right":False})

    def save(fig, name):
        path = figures/name
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(path) as picture:
            picture.verify()

    def table_png(name, title, subtitle, headers, rows, highlight=(), note=""):
        height = max(3.7, 1.8 + .34*(len(rows)+1))
        fig = plt.figure(figsize=(14.5,height), facecolor="white")
        fig.text(.04, 1-.22/height, title, fontsize=18, weight="bold", va="top", color="#14213d")
        fig.text(.04, 1-.64/height, subtitle, fontsize=10, va="top", color="#475569")
        ax = fig.add_axes([.035,.62/height,.93,(height-1.7)/height])
        ax.axis("off")
        artist = ax.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0,0,1,1])
        artist.auto_set_font_size(False)
        artist.set_fontsize(10)
        for (r,c),cell in artist.get_celld().items():
            cell.set_edgecolor("white")
            if r == 0:
                cell.set_facecolor("#14213d")
                cell.set_text_props(color="white",weight="bold",fontsize=9.5)
            else:
                cell.set_facecolor("#eef3f9" if r%2 else "white")
                if r-1 in highlight:
                    cell.set_facecolor("#dcecff")
                    cell.set_text_props(weight="bold")
        fig.text(.04,.24/height,note,fontsize=9,color="#475569",va="bottom")
        save(fig,name)

    test_rows = [[r["metric"].title(),r["k"],f"{float(index[r['metric'],int(r['k'])]['accuracy_mean'])*100:.2f}",
                  *[f"{float(r[s])*100:.2f}" for s in SCORES]] for r in tests]
    table_png("01_test_results.png", "Uniform voting + nearest tie | Official test results",
              "All odd k from 1 to 101 searched by 5-fold CV | Full 50,000 train / 10,000 test images",
              ["Distance","Selected k","CV Accuracy\n(%)","Test Accuracy\n(%)","Macro Precision\n(%)","Macro Recall\n(%)","Macro F1\n(%)"],test_rows,
              highlight=[i for i,r in enumerate(tests) if r["selected_within_voting"]=="True"],
              note="Every neighbor has one vote. A tied winning class is chosen by its closest neighbor; equal distances use stable training-index order.")
    table_png("02_cv_accuracy_k1_to_101.png", "All 51 odd k values | 5-fold CV accuracy",
              "Uniform majority vote; nearest neighbor in tied winning classes | Mean ± sample SD (%)",
              ["k",*[m.title()+"\nAccuracy (%)" for m in METRICS]],
              [[k,*[f"{float(index[m,k]['accuracy_mean'])*100:.2f} ± {float(index[m,k]['accuracy_std'])*100:.2f}" for m in METRICS]] for k in ks],
              highlight=[ks.index(best["k"])], note="Each fold: 40,000 training and 10,000 validation images. All 50,000 training images are validated once.")
    colors = {"euclidean":"#2563eb","manhattan":"#dc2626","cosine":"#059669"}
    fig,axes = plt.subplots(2,2,figsize=(13,9.5),constrained_layout=True)
    for ax,score,label in zip(axes.flat,SCORES,["Accuracy","Macro Precision","Macro Recall","Macro F1"]):
        for metric in METRICS:
            means = np.array([float(index[metric,k][score+"_mean"])*100 for k in ks])
            stds = np.array([float(index[metric,k][score+"_std"])*100 for k in ks])
            ax.plot(ks,means,label=metric.title(),color=colors[metric],linewidth=1.8)
            ax.fill_between(ks,means-stds,means+stds,color=colors[metric],alpha=.12)
        ax.set(title=label,xlabel="k",ylabel=label+" (%)")
        ax.set_xticks([1,11,21,31,41,51,61,71,81,91,101])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Uniform majority + nearest tie | 5-fold CV | k = 1, 3, ..., 101",fontsize=16)
    save(fig,"03_cv_curves.png")
    cm = read(root/"confusion_uniform_nearest.csv")
    matrix = np.array([[int(r[c]) for c in CLASS_NAMES] for r in cm])
    fig,ax = plt.subplots(figsize=(11,9.5),constrained_layout=True)
    pic = ax.imshow(matrix,cmap="Blues",vmin=0)
    for r in range(10):
        for c in range(10):
            ax.text(c,r,str(matrix[r,c]),ha="center",va="center",fontsize=10,
                    color="white" if matrix[r,c]>matrix.max()*.5 else "#14213d")
    ax.set_xticks(range(10),CLASS_NAMES,rotation=40,ha="right")
    ax.set_yticks(range(10),CLASS_NAMES)
    ax.set(xlabel="Predicted class",ylabel="True class",title=f"Uniform + nearest tie | {best['metric']}, k={best['k']} | Official test")
    fig.colorbar(pic,ax=ax,label="Images",shrink=.85)
    save(fig,"04_confusion_matrix.png")
    for number,metric in enumerate(METRICS,5):
        selected_k = metadata["per_metric_selected_k"]["uniform_nearest"][metric]
        rows = [[k,*[f"{float(index[metric,k][s+'_mean'])*100:.2f} ± {float(index[metric,k][s+'_std'])*100:.2f}" for s in SCORES]] for k in ks]
        table_png(f"{number:02d}_cv_{metric}.png", f"{metric.title()} | All 51 k settings",
                  "Uniform voting + nearest tie | 5-fold CV mean ± sample standard deviation (%)",
                  ["k","Accuracy (%)","Macro Precision (%)","Macro Recall (%)","Macro F1 (%)"],rows,
                  highlight=[ks.index(selected_k)],note="Blue row: selected by mean CV accuracy. Standard deviation uses ddof=1; it is not a confidence interval.")
    table_png("08_classification_report.png",f"Selected model | {best['metric'].title()}, k={best['k']}",
              "Uniform voting + nearest tie | Official test | 1,000 true images per class",
              ["Class","Precision (%)","Recall (%)","F1 (%)","Support"],
              [[r["class_name"],*[f"{float(r[s])*100:.2f}" for s in ["precision","recall","f1-score"]],int(float(r["support"]))] for r in classes],
              note="Class-specific metrics. Overall precision, recall and F1 use macro averaging.")

    def html_table(rows,columns,formatter=None):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(c)}</th>" for c in columns)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(formatter(r,c) if formatter else str(r[c]))}</td>" for c in columns)+"</tr>" for r in rows)+"</tbody></table>"

    def pct(r,c):
        return f"{float(r[c])*100:.2f}%" if c in SCORES or c.endswith("_accuracy") or c in ("precision","recall","f1-score") else str(r[c])

    display = [dict(metric=r["metric"],k=r["k"],**{s:f"{float(r[s+'_mean'])*100:.2f} ± {float(r[s+'_std'])*100:.2f}%" for s in SCORES}) for r in summary]
    chosen_test = next(r for r in tests if r["selected_within_voting"]=="True")
    report = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>다수결·최근접 동률 처리 KNN: k=101까지</title><style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;line-height:1.7;color:#14213d}}table{{border-collapse:collapse;font-size:14px;margin:20px 0}}th,td{{border:1px solid #d9e1ec;padding:8px 10px;text-align:right}}th{{background:#edf2f8}}td:first-child{{text-align:left}}img{{max-width:100%}}.result{{background:#e8f1ff;padding:16px}}</style>
<h1>다수결 + 가까운 이웃 동률 처리: k=1,3,…,101</h1>
<p class="result">CV 최적 모델: <strong>{best['metric']}, k={best['k']}</strong>.<br>
CV Accuracy {best['accuracy_mean']:.2%} ± {best['accuracy_std']*100:.2f}%p,
공식 테스트 Accuracy <strong>{float(chosen_test['accuracy']):.2%}</strong>.</p>
<h2>설정과 검증</h2><p>학습 50,000장 전체, 공식 테스트 10,000장 전체, 원본 3,072차원 픽셀을 사용했다.
각 이웃은 동일한 1표를 가지며, 최다 득표 클래스들이 동률이면 그 후보들 중 가장 가까운 이웃의 클래스를 선택했다.
거리까지 같으면 안정 정렬된 학습 인덱스 순서를 따른다. 거리 가중 투표는 사용하지 않았다.</p>
<p>k는 1부터 101까지의 모든 홀수 {len(ks)}개이다. Euclidean·Manhattan·Cosine의 3개 거리와 결합해 {len(summary)}개 설정을 평가했다.
Stratified 5-fold, shuffle=True, random_state={metadata['seed']}를 사용해 매 fold 40,000장 학습 및 10,000장 검증을 수행했다.
총 {len(summary)*5}개 fold 결과를 계산했으며, 각 이미지는 정확히 한 번 검증됐다.
기존 데이터 해시와 fold 분할이 동일하고, 기존 k 값에서 일반 투표 예측이 재현됨을 확인했다.</p>
<p>평균 CV 정확도로 거리별 k 및 최종 거리를 선택한 다음 테스트 예측 전에 metadata.json에 저장했다.
동률 설정은 작은 k, 거리 이름 순서로 선택했다. 최종 테스트에서는 학습 50,000장을 모두 사용했다.</p>
<h2>공식 테스트 결과</h2>{html_table(tests,['metric','k','selected_within_voting',*SCORES],pct)}
<img src="figures/01_test_results.png">
<h2>기존 k 범위와 비교</h2>{html_table(comparisons,['metric','previous_k','previous_cv_accuracy','previous_test_accuracy','new_k','new_cv_accuracy','new_test_accuracy'],pct)}
<p>기존 범위는 k=1,3,5,7,9,11,15,21이었다. 위 비교는 동일한 다수결·최근접 동률 처리 방식의 결과이다.</p>
<h2>전체 k 곡선</h2><img src="figures/03_cv_curves.png">
<p>음영은 fold별 표본 표준편차이며 신뢰구간이 아니다. 정밀도·재현율·F1은 클래스별 지표의 macro 평균이다.</p>
<h2>전체 {len(summary)}개 설정의 CV 지표</h2>{html_table(display,['metric','k',*SCORES])}
<h2>최종 모델의 클래스별 결과</h2>{html_table(classes,['class_name','precision','recall','f1-score','support'],pct)}
<img src="figures/04_confusion_matrix.png">
<h2>재현 및 다운로드</h2><p>실행 시간: {metadata['elapsed_seconds']:.1f}초. 단일 {metadata['device_name']} 사용.</p>
<p><code>{html.escape(metadata['command'])}</code></p>
<p><a href="cv_summary.csv">전체 CV 요약</a> · <a href="cv_folds.csv">전체 fold 결과</a> · <a href="test_results.csv">테스트 결과</a> ·
<a href="range_comparison.csv">기존 범위 비교</a> · <a href="metadata.json">메타데이터</a> · <a href="nearest_k101_results.zip">보고서·표·그래프·CSV ZIP</a></p></html>"""
    (root/"report.html").write_text(report,encoding="utf-8")
    with zipfile.ZipFile(root/"nearest_k101_results.zip","w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [root/"report.html",root/"metadata.json",*root.glob("*.csv"),*figures.glob("*.png")]:
            bundle.write(path,str(path.relative_to(root)))
    with zipfile.ZipFile(root/"nearest_k101_results.zip") as bundle:
        assert bundle.testzip() is None
    print("REPORT",root/"report.html")
    print("IMAGES",len(list(figures.glob("*.png"))),"300dpi PNGs")
    print("COMPARISON",json.dumps(comparisons,ensure_ascii=False))


if __name__ == "__main__":
    main()
