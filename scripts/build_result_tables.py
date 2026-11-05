"""Render saved CSV metrics as GitHub tables; never rerun KNN or edit CSVs."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
SCORE_NAMES = ("정확도", "정밀도", "재현율", "F1")
DISTANCES = {"euclidean": "Euclidean", "manhattan": "Manhattan", "cosine": "Cosine"}
VOTING = {
    "uniform_class_id": "다수결 + 작은 클래스 번호",
    "uniform_nearest": "다수결 + 최근접 동률 해결",
    "distance_weighted": "거리 역수 가중 투표",
}
TITLES = {
    "01_baseline": "기본 KNN",
    "02_tie_analysis": "동률 처리 영향 분석",
    "03_voting_comparison": "투표 방식 3종 비교",
    "04_k101": "k=101까지 확장",
}
CLASS_KO = dict(zip(
    ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
    ("비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"),
))
NOTE = "단위는 %입니다. 정밀도·재현율·F1은 Macro 평균이며, CV는 5개 fold의 평균 ± 표본 표준편차(ddof=1)입니다."


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def pct(value):
    return f"{float(value) * 100:.2f}"


def meanstd(row, score):
    return f"{pct(row[score + '_mean'])} ± {pct(row[score + '_std'])}"


def vote_groups(rows):
    votes = list(dict.fromkeys(row.get("voting", "uniform_class_id") for row in rows))
    return [(vote, [r for r in rows if r.get("voting", "uniform_class_id") == vote]) for vote in votes]


def chosen(row):
    return row.get("selected_by_cv", row.get("selected_within_voting")) == "True"


def emphasize(row):
    return [f"**{cell}**" for cell in row]


def write(directory, name, title, paragraphs):
    (directory / name).write_text("\n\n".join(["# " + title, *paragraphs]) + "\n", encoding="utf-8")


def render_cv(directory, rows):
    content = [NOTE, f"전체 **{len(rows)}개 설정**입니다. 원자료: [cv_summary.csv](cv_summary.csv)."]
    for vote, group in vote_groups(rows):
        content.append("## " + VOTING[vote])
        best = min(group, key=lambda r: (-float(r["accuracy_mean"]), int(r["k"]), r["metric"]))
        displayed = []
        for row in group:
            cells = [DISTANCES[row["metric"]], row["k"], *[meanstd(row, s) for s in SCORES]]
            displayed.append(emphasize(cells) if row is best else cells)
        content.append(table(["거리", "k", *SCORE_NAMES], displayed))
    write(directory, "cv_results.md", "전체 5-fold CV 결과", content)


def render_folds(directory, rows):
    content = [
        "단위는 %입니다. 각 fold는 학습 40,000장 / 검증 10,000장입니다. 정밀도·재현율·F1은 Macro 평균입니다.",
        f"전체 **{len(rows)}개 fold 결과**입니다. 원자료: [cv_folds.csv](cv_folds.csv).",
        "검증 클래스별 이미지 수가 1,000장으로 같으므로 Macro 재현율과 정확도는 같습니다.",
    ]
    for vote, group in vote_groups(rows):
        content.append("## " + VOTING[vote])
        group = sorted(group, key=lambda r: (list(DISTANCES).index(r["metric"]), int(r["k"]), int(r["fold"])))
        content.append(table(["거리", "k", "Fold", *SCORE_NAMES], [
            [DISTANCES[r["metric"]], r["k"], r["fold"], *[pct(r[s]) for s in SCORES]] for r in group
        ]))
    write(directory, "fold_results.md", "전체 fold별 결과", content)


def render_test(directory, rows):
    content = [
        "단위는 %입니다. 정밀도·재현율·F1은 Macro 평균입니다.",
        "거리별 k는 평균 CV 정확도로 선택했습니다. 학습 50,000장 전체를 사용해 공식 테스트 10,000장을 평가했습니다. 굵은 행은 해당 투표 방식에서 CV로 선택한 최종 모델입니다.",
        "원자료: [test_results.csv](test_results.csv).",
    ]
    for vote, group in vote_groups(rows):
        content.append("## " + VOTING[vote])
        displayed = []
        for r in group:
            count = round(float(r["accuracy"]) * int(r["n_test"]))
            cells = [DISTANCES[r["metric"]], r["k"], *[pct(r[s]) for s in SCORES], f"{count:,} / {int(r['n_test']):,}"]
            displayed.append(emphasize(cells) if chosen(r) else cells)
        content.append(table(["거리", "k", *SCORE_NAMES, "정답 수"], displayed))
    write(directory, "test_results.md", "공식 테스트 결과", content)


def render_classes(directory, rows):
    content = ["각 투표 방식에서 CV로 선택한 모델의 테스트 결과입니다. 단위는 %이며, 클래스마다 테스트 이미지가 1,000장입니다."]
    for vote, group in vote_groups(rows):
        content.append("## " + VOTING[vote])
        if "metric" in group[0]:
            content.append(f"선택 모델: {DISTANCES[group[0]['metric']]}, k={group[0]['k']}.")
        displayed = [[CLASS_KO[r["class_name"]] + " — " + r["class_name"],
                      *[pct(r[s]) for s in ("precision", "recall", "f1-score")],
                      f"{int(float(r['support'])):,}"] for r in group]
        displayed.append(emphasize(["Macro 평균", *[
            pct(sum(float(r[s]) for r in group) / len(group)) for s in ("precision", "recall", "f1-score")
        ], f"{sum(int(float(r['support'])) for r in group):,}"]))
        content.append(table(["클래스", "정밀도", "재현율", "F1", "이미지 수"], displayed))
    content.append("원자료: [classification_report.csv](classification_report.csv). Macro F1은 클래스별 F1의 평균입니다.")
    write(directory, "class_results.md", "클래스별 테스트 결과", content)


def render_confusions(directory):
    content = ["행은 실제 클래스, 열은 예측 클래스이며 단위는 이미지 수입니다. 대각선은 정답 수입니다."]
    for path in sorted(directory.glob("confusion*.csv")):
        vote = path.stem.removeprefix("confusion_")
        if vote == "matrix":
            vote = "uniform_class_id"
        content.append("## " + VOTING[vote])
        displayed = []
        for row in read(path):
            displayed.append([CLASS_KO[row["actual"]], *[
                f"**{row[c]}**" if row["actual"] == c else row[c] for c in CLASS_KO
            ]])
        content.append(table(["실제 ↓ / 예측 →", *CLASS_KO.values()], displayed))
        content.append(f"원자료: [{path.name}]({path.name}).")
    write(directory, "confusion_matrices.md", "혼동행렬", content)


def render_ties(directory):
    rows = read(directory / "tie_summary.csv")
    write(directory, "tie_results.md", "동률 처리 영향", [
        "Manhattan 거리, 학습 50,000장의 OOF 예측을 분석했습니다. 이 분석에서는 공식 테스트를 사용하지 않았습니다. 정확도·동률 비율 단위는 %이며, 정확도 차이는 %p입니다.",
        table(["k", "동률 이미지 수", "동률 비율", "작은 클래스 번호 CV 정확도", "최근접 동률 해결 CV 정확도", "정확도 변화(%p)", "오답→정답", "정답→오답"], [
            [r["k"], f"{int(r['tied_queries']):,}", pct(r["tie_fraction"]), pct(r["original_accuracy"]),
             pct(r["nearest_class_tiebreak_accuracy"]), f"{(float(r['nearest_class_tiebreak_accuracy'])-float(r['original_accuracy']))*100:+.2f}",
             r["corrected_by_tiebreak"], r["harmed_by_tiebreak"]] for r in rows
        ]),
        "원자료: [tie_summary.csv](tie_summary.csv). 모든 fold 크기가 같아 통합 OOF 정확도와 fold 평균 정확도는 같습니다.",
    ])
    alternatives = directory / "alternative_cv_summary.csv"
    if alternatives.exists():
        alt = read(alternatives)
        write(directory, "cv_results.md", "최근접 동률 해결의 CV 지표", [
            NOTE, "Manhattan 거리에서 기존 8개 k를 평가한 결과입니다. 원자료: [alternative_cv_summary.csv](alternative_cv_summary.csv).",
            table(["거리", "k", *SCORE_NAMES], [["Manhattan", r["k"], *[meanstd(r,s) for s in SCORES]] for r in alt]),
        ])
    folds = read(directory / "tie_folds.csv")
    write(directory, "fold_results.md", "동률 분석의 fold별 결과", [
        "Manhattan 거리의 8개 k × 5개 fold = 40개 결과입니다. 이미지 수를 제외한 수치는 %입니다.",
        table(["k", "Fold", "검증 이미지 수", "동률 이미지 수", "작은 클래스 번호 정확도", "최근접 동률 해결 정확도"], [
            [r["k"], r["fold"], r["n_queries"], r["ties"], pct(r["original_accuracy"]), pct(r["nearest_class_tiebreak_accuracy"])]
            for r in sorted(folds,key=lambda r:(int(r["k"]),int(r["fold"])))
        ]), "원자료: [tie_folds.csv](tie_folds.csv).",
    ])
    links = ["[동률 발생·성능 변화](tie_results.md)", "[전체 fold 결과](fold_results.md)"]
    if alternatives.exists():
        links.append("[4개 CV 지표](cv_results.md)")
    content = [
        "같은 Manhattan 이웃 목록에 작은 클래스 번호 선택과 최근접 동률 해결을 각각 적용해 비교했습니다. 기존 8개 k를 분석했습니다.",
        "k=5에서 동률은 **21.86%**, CV 정확도는 **36.30% → 37.85%**였습니다. 이 분석의 평가는 CV 전용입니다.",
        " · ".join(links),
        "[실행 코드](../../experiments/02_tie_analysis/run.py) · [실험 목록](../README.md)",
    ]
    if (directory / "voting_tie_analysis.png").exists():
        content.append("![동률 처리 분석](voting_tie_analysis.png)")
    write(directory, "README.md", "02. 동률 처리 영향 분석", content)


def render_comparison(directory):
    path = directory / "voting_comparison.csv"
    if path.exists():
        rows = read(path)
        write(directory, "comparison.md", "투표 방식별 CV 최적 모델", [
            NOTE,
            table(["투표 방식", "거리", "k", "CV 정확도", "테스트 정확도", "테스트 정밀도", "테스트 재현율", "테스트 F1"], [
                [VOTING[r["voting"]], DISTANCES[r["metric"]], r["k"], f"{pct(r['cv_accuracy'])} ± {pct(r['cv_accuracy_std'])}", *[pct(r[s]) for s in SCORES]] for r in rows
            ]), "원자료: [voting_comparison.csv](voting_comparison.csv).",
        ])
    path = directory / "range_comparison.csv"
    if path.exists():
        rows = read(path)
        write(directory, "range_comparison.md", "k 탐색 범위 확장 전후 비교", [
            "동일한 다수결 + 최근접 동률 해결 방식입니다. 이전 범위는 1,3,5,7,9,11,15,21이며, 확장 범위는 1부터 101까지의 모든 홀수입니다. 정확도 단위는 %입니다.",
            table(["거리", "기존 최적 k", "기존 CV 정확도", "기존 테스트 정확도", "확장 최적 k", "확장 CV 정확도", "확장 테스트 정확도"], [
                [DISTANCES[r["metric"]], r["previous_k"], pct(r["previous_cv_accuracy"]), pct(r["previous_test_accuracy"]),
                 r["new_k"], pct(r["new_cv_accuracy"]), pct(r["new_test_accuracy"])] for r in rows
            ]), "탐색 범위를 확장해도 세 거리의 최적 k=5와 테스트 예측은 동일했습니다.",
        ])


def render_main(directory):
    summary = read(directory / "cv_summary.csv")
    folds = read(directory / "cv_folds.csv")
    tests = read(directory / "test_results.csv")
    render_cv(directory, summary)
    render_folds(directory, folds)
    render_test(directory, tests)
    render_classes(directory, read(directory / "classification_report.csv"))
    render_confusions(directory)
    render_comparison(directory)
    links = " · ".join(f"[{label}]({name})" for label,name in [
        ("전체 CV 결과", "cv_results.md"), ("전체 fold 결과", "fold_results.md"),
        ("테스트 결과", "test_results.md"), ("클래스별 결과", "class_results.md"),
        ("혼동행렬", "confusion_matrices.md"),
    ])
    brief = {
        "01_baseline": "모든 이웃에 동일한 1표를 부여하고, 최다 득표가 동률이면 가장 작은 클래스 번호를 선택했습니다. k는 기존 8개 값입니다.",
        "03_voting_comparison": "세 투표 방식을 동일한 이웃·거리·k·fold 조건에서 비교했습니다. 각 방식마다 3개 거리 × 8개 k를 평가했습니다.",
        "04_k101": "다수결 + 최근접 동률 해결 방식으로 k=1,3,…,101의 모든 홀수 51개를 평가했습니다. 거리 가중 투표의 확장은 수행하지 않았습니다.",
    }
    images = {
        "01_baseline": "cross_validation.png",
        "03_voting_comparison": "figures/04_cv_voting_curves.png",
        "04_k101": "figures/03_cv_curves.png",
    }
    displayed = []
    for row in tests:
        if not chosen(row):
            continue
        vote = row.get("voting", "uniform_class_id")
        cv = next(r for r in summary if r["metric"] == row["metric"] and r["k"] == row["k"] and r.get("voting", "uniform_class_id") == vote)
        displayed.append([VOTING[vote], DISTANCES[row["metric"]], row["k"], meanstd(cv,"accuracy"), *[pct(row[s]) for s in SCORES]])
    content = [brief[directory.name], NOTE,
               f"전체 **{len(summary)}개 설정**, **{len(folds)}개 fold 결과**, 거리별 CV 최적 모델의 **{len(tests)}개 테스트 결과**입니다.",
               table(["투표 방식", "거리", "k", "CV 정확도", "테스트 정확도", "테스트 정밀도", "테스트 재현율", "테스트 F1"],displayed), links]
    if (directory / "comparison.md").exists():
        content.append("[투표 방식별 최적 모델 비교](comparison.md)")
    if (directory / "range_comparison.md").exists():
        content.append("[탐색 범위 확장 전후 비교](range_comparison.md)")
    if (directory / images[directory.name]).exists():
        content.append(f"![CV 성능 곡선]({images[directory.name]})")
    content.append(f"[실행 코드](../../experiments/{directory.name}/run.py) · [실험 목록](../README.md)")
    write(directory,"README.md",directory.name[:2]+". "+TITLES[directory.name],content)


def render_index(root):
    entries = []
    for experiment in ["01_baseline","03_voting_comparison","04_k101"]:
        directory = root / experiment
        cv_rows = read(directory / "cv_summary.csv")
        for r in read(directory / "test_results.csv"):
            if not chosen(r):
                continue
            vote = r.get("voting","uniform_class_id")
            # The baseline is reproduced within experiment 03; show it once.
            if experiment == "03_voting_comparison" and vote == "uniform_class_id":
                continue
            cv = next(c for c in cv_rows if c["metric"] == r["metric"] and c["k"] == r["k"] and c.get("voting","uniform_class_id") == vote)
            entries.append(dict(experiment=experiment,voting=vote,metric=r["metric"],k=r["k"],cv_accuracy_mean=cv["accuracy_mean"],cv_accuracy_std=cv["accuracy_std"],**{s:r[s] for s in SCORES}))
    with (root / "comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    display = [[f"[{r['experiment'][:2]}]({r['experiment']}/README.md)", VOTING[r["voting"]],DISTANCES[r["metric"]],r["k"],
                f"{pct(r['cv_accuracy_mean'])} ± {pct(r['cv_accuracy_std'])}",*[pct(r[s]) for s in SCORES]] for r in entries]
    write(root,"README.md","실험별 결과",[
        NOTE,
        "주실험은 전체 학습 50,000장에 Stratified 5-fold 검증을 적용했습니다. 최종 모델은 평균 CV 정확도로 선택한 뒤 전체 학습 데이터로 공식 테스트 10,000장을 평가했습니다.",
        table(["실험","투표 방식","거리","k","CV 정확도","테스트 정확도","테스트 정밀도","테스트 재현율","테스트 F1"],display),
        "최고 관측 테스트 정확도는 **거리 역수 가중 투표, Manhattan, k=5의 39.50%**입니다. 최신 k 확장 실험은 **다수결 + 최근접 동률 해결, Manhattan, k=5의 39.21%**입니다. 각 모델의 거리와 k는 CV로 선택했습니다.",
        "## 전체 표 바로가기",
        table(["실험","설정 수","fold 결과 수","결과 문서"],[
            ["01 기본 KNN",24,120,"[결과](01_baseline/README.md) · [CV](01_baseline/cv_results.md) · [Fold](01_baseline/fold_results.md) · [테스트](01_baseline/test_results.md)"],
            ["02 동률 분석",8,40,"[결과](02_tie_analysis/README.md) · [동률](02_tie_analysis/tie_results.md) · [CV](02_tie_analysis/cv_results.md) · [Fold](02_tie_analysis/fold_results.md)"],
            ["03 투표 방식 비교",72,360,"[결과](03_voting_comparison/README.md) · [CV](03_voting_comparison/cv_results.md) · [Fold](03_voting_comparison/fold_results.md) · [테스트](03_voting_comparison/test_results.md)"],
            ["04 k 범위 확장",153,765,"[결과](04_k101/README.md) · [CV](04_k101/cv_results.md) · [Fold](04_k101/fold_results.md) · [테스트](04_k101/test_results.md)"],
        ]),
        "02는 Manhattan CV 전용 분석입니다. 03은 세 투표 방식의 결과를 포함하므로 01의 기본 투표 결과가 재현되어 있습니다. 위 설정 수는 실험별 기록 수이며, 중복을 제거한 고유 모델 수를 의미하지 않습니다.",
        "## 결과 해석",
        "- 기본 동률 규칙에서는 최적 k=1이었지만, 최근접 동률 해결과 거리 가중 투표에서는 최적 k=5였습니다. 같은 k라도 투표 규칙에 따라 결과가 달라집니다.\n- k=101까지 확장해도 최근접 동률 해결 방식의 최적 설정은 바뀌지 않았습니다. k가 커질수록 정확도는 전반적으로 감소했으나 모든 인접 k에서 단조 감소한 것은 아닙니다.\n- 원시 픽셀은 배경·위치·밝기의 영향을 받습니다. 큰 k에서 서로 다른 클래스가 섞이는 현상은 성능 감소의 가능한 해석이며, 원인을 직접 검증한 추가 실험은 수행하지 않았습니다.",
        "## 파일과 재현",
        "CSV는 원본 수치(0~1)를 보존하고, Markdown 표는 이를 백분율로 변환해 소수 둘째 자리까지 표시합니다. metadata.json은 최초 실행 기록이며 당시 로컬 경로도 보존했습니다. [원본 경로·코드 변경 내역](../docs/PROVENANCE.md)을 참고하세요.",
        "저장된 결과에는 CSV·메타데이터·PNG를 포함했습니다. 데이터셋과 NPZ 이웃·예측 캐시는 업로드에서 제외했습니다. 새 실험은 [실행 안내](../README.md#실험-재현)에 따라 runs/에 생성하며, 02~04 실행 전 01을 실행해 predictions.npz를 생성해야 합니다.",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",type=Path,default=ROOT/"results")
    args = parser.parse_args()
    root = args.results_root.resolve()
    for name in TITLES:
        directory = root / name
        if name == "02_tie_analysis":
            render_ties(directory)
        else:
            render_main(directory)
    render_index(root)
    print(f"Generated {len(list(root.rglob('*.md')))} Markdown documents from saved CSVs: {root}")


if __name__ == "__main__":
    main()
