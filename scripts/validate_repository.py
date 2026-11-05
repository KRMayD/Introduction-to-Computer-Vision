"""Validate published result numbers, copied files and local Markdown links."""

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SCORES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-10):
        raise AssertionError(f"Metric mismatch: {actual} != {expected}")


def vote(row):
    return row.get("voting", "uniform_class_id")


def selected(row):
    return row.get("selected_by_cv", row.get("selected_within_voting")) == "True"


def key(row):
    return -float(row["accuracy_mean"]), int(row["k"]), row["metric"]


def markdown_rows(path):
    lines = [line for line in path.read_text().splitlines() if line.startswith("| ")]
    separators = sum(bool(re.fullmatch(r"[| :\-]+", line)) for line in lines)
    return len(lines) - 2 * separators


def validate_experiment(name, settings, nfolds, ntests):
    root = ROOT / "results" / name
    summary, folds, tests = (read(root / n) for n in ("cv_summary.csv", "cv_folds.csv", "test_results.csv"))
    assert (len(summary),len(folds),len(tests)) == (settings,nfolds,ntests)
    if name == "04_k101":
        assert sorted({int(r["k"]) for r in summary}) == list(range(1,102,2))
        assert {vote(r) for r in summary} == {"uniform_nearest"}
    else:
        assert sorted({int(r["k"]) for r in summary}) == [1,3,5,7,9,11,15,21]
    assert len({(vote(r),r["metric"],r["k"]) for r in summary}) == len(summary)
    for row in summary:
        group = [f for f in folds if (vote(f),f["metric"],f["k"]) == (vote(row),row["metric"],row["k"])]
        assert {f["fold"] for f in group} == {"1","2","3","4","5"}
        assert len(group) == 5
        for score in SCORES:
            values = [float(f[score]) for f in group]
            close(statistics.mean(values),row[score+"_mean"])
            close(statistics.stdev(values),row[score+"_std"])
    for row in folds:
        assert (int(row["n_train"]),int(row["n_valid"])) == (40000,10000)
        close(row["accuracy"],row["recall_macro"])
    for row in tests:
        assert (int(row["n_train"]),int(row["n_test"])) == (50000,10000)
        candidates = [s for s in summary if vote(s) == vote(row) and s["metric"] == row["metric"]]
        assert int(row["k"]) == int(min(candidates,key=key)["k"])
        overall = min([s for s in summary if vote(s) == vote(row)],key=key)
        assert selected(row) == (row["metric"] == overall["metric"])
        close(float(row["accuracy"])*10000,round(float(row["accuracy"])*10000))
    classes = read(root / "classification_report.csv")
    for path in root.glob("confusion*.csv"):
        voting = path.stem.removeprefix("confusion_")
        if voting == "matrix":
            voting = "uniform_class_id"
        records = read(path)
        assert [r["actual"] for r in records] == list(CLASSES)
        matrix = [[int(r[c]) for c in CLASSES] for r in records]
        assert all(sum(row) == 1000 for row in matrix)
        target = next(t for t in tests if vote(t) == voting and selected(t))
        group = [c for c in classes if vote(c) == voting]
        assert [c["class_name"] for c in group] == list(CLASSES)
        precision, recall, f1 = [], [], []
        for i, row in enumerate(group):
            tp = matrix[i][i]
            predicted = sum(r[i] for r in matrix)
            true = sum(matrix[i])
            p = tp/predicted if predicted else 0
            r = tp/true if true else 0
            f = 2*tp/(predicted+true) if predicted+true else 0
            close(row["precision"],p)
            close(row["recall"],r)
            close(row["f1-score"],f)
            assert float(row["support"]) == true
            precision.append(p)
            recall.append(r)
            f1.append(f)
        close(target["accuracy"],sum(matrix[i][i] for i in range(10))/10000)
        for field,values in [("precision_macro",precision),("recall_macro",recall),("f1_macro",f1)]:
            close(target[field],statistics.mean(values))
    assignment = read(root/"fold_assignments.csv")
    assert len(assignment) == 50000
    assert [int(r["train_index"]) for r in assignment] == list(range(50000))
    assert Counter(r["validation_fold"] for r in assignment) == {str(i):10000 for i in range(1,6)}
    assert set(Counter((r["validation_fold"],r["label"]) for r in assignment).values()) == {1000}
    assert markdown_rows(root/"cv_results.md") == settings
    assert markdown_rows(root/"fold_results.md") == nfolds
    assert markdown_rows(root/"test_results.md") == ntests
    assert markdown_rows(root/"class_results.md") == len(classes) + len({vote(r) for r in classes})
    assert markdown_rows(root/"confusion_matrices.md") == 10*len(list(root.glob("confusion*.csv")))
    return summary


def validate_ties(base_summary, voting_summary):
    root = ROOT / "results/02_tie_analysis"
    rows,folds,alt = (read(root/p) for p in ["tie_summary.csv","tie_folds.csv","alternative_cv_summary.csv"])
    assert (len(rows),len(folds),len(alt)) == (8,40,8)
    for row in rows:
        group = [f for f in folds if f["k"] == row["k"]]
        assert len(group) == 5 and {r["fold"] for r in group} == {"1","2","3","4","5"}
        assert int(row["tied_queries"]) == sum(int(r["ties"]) for r in group)
        assert int(row["n_queries"]) == 50000
        close(row["tie_fraction"],int(row["tied_queries"])/50000)
        close(row["original_accuracy"],statistics.mean(float(f["original_accuracy"]) for f in group))
        close(row["nearest_class_tiebreak_accuracy"],statistics.mean(float(f["nearest_class_tiebreak_accuracy"]) for f in group))
        close((float(row["nearest_class_tiebreak_accuracy"])-float(row["original_accuracy"]))*50000,
              int(row["corrected_by_tiebreak"])-int(row["harmed_by_tiebreak"]))
        base = next(b for b in base_summary if b["metric"] == "manhattan" and b["k"] == row["k"])
        close(base["accuracy_mean"],row["original_accuracy"])
    for row in alt:
        current = next(r for r in voting_summary if r["metric"] == "manhattan" and r["k"] == row["k"] and vote(r) == "uniform_nearest")
        for score in SCORES:
            close(row[score+"_mean"],current[score+"_mean"])
            close(row[score+"_std"],current[score+"_std"])
    for name,count in [("tie_results.md",8),("cv_results.md",8),("fold_results.md",40)]:
        assert markdown_rows(root/name) == count


def validate_links():
    links = 0
    for document in ROOT.rglob("*.md"):
        if ".git" in document.parts:
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            url = urlsplit(target)
            if url.scheme or target.startswith("#"):
                continue
            path = (document.parent / unquote(url.path)).resolve()
            assert path.exists(), f"Broken link in {document.relative_to(ROOT)}: {target}"
            if url.fragment and path.suffix == ".md":
                headings = re.findall(r"^#+\s+(.+)$",path.read_text(),flags=re.MULTILINE)
                slugs = {re.sub(r"[^\w\- ]","",h.lower()).replace(" ","-") for h in headings}
                assert unquote(url.fragment) in slugs, f"Broken anchor: {document}: {target}"
            links += 1
    return links


def main():
    manifest = json.loads((ROOT/"docs/source_manifest.json").read_text())
    for row in manifest:
        path = ROOT/row["organized"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["organized_sha256"], path
        if row["kind"] == "result":
            assert row["original_sha256"] == row["organized_sha256"]
    base = validate_experiment("01_baseline",24,120,3)
    voting = validate_experiment("03_voting_comparison",72,360,9)
    latest = validate_experiment("04_k101",153,765,3)
    validate_ties(base,voting)
    original_votes = {(vote(r),r["metric"],r["k"]):r for r in voting}
    for row in latest:
        old = original_votes.get((vote(row),row["metric"],row["k"]))
        if old:
            for score in SCORES:
                close(old[score+"_mean"],row[score+"_mean"])
                close(old[score+"_std"],row[score+"_std"])
    hashes = {hashlib.sha256((ROOT/"results"/name/"fold_assignments.csv").read_bytes()).hexdigest()
              for name in ["01_baseline","03_voting_comparison","04_k101"]}
    assert len(hashes) == 1
    links = validate_links()
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    assert not any(p.suffix in {".tar",".gz",".npz",".npy",".zip"} for p in files)
    assert max(p.stat().st_size for p in files) < 20*1024*1024
    print(f"PASS: {len(manifest)} source hashes, 249 main CV settings, 8 tie settings, 1,285 fold records, 15 test rows, class metrics and confusion matrices.")
    print(f"PASS: identical fold assignments, old/new shared settings, complete Markdown table row counts, {links} local links.")
    print(f"PASS: {len(files)} repository files; no dataset archives, NPZ caches or ZIPs; all files under 20 MiB.")


if __name__ == "__main__":
    main()
