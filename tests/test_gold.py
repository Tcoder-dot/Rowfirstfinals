"""Exam A/B/C against SciPy gold. Run: python3 tests/test_gold.py"""
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import _send_analysis
from chapter4 import write_docx
from charts import make_charts
from handle import handle_analyze


def near(a, b, tol=0.02):
    return abs(a - b) <= tol


def chart_regression_failures():
    """Protect the chart type chosen for each supported reporting outcome."""
    group = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    paired = handle_analyze({
        "text": "id,before,after\n1,72,78\n2,75,79\n3,70,74\n4,80,84\n5,68,73",
    })
    regression = handle_analyze({
        "text": "hours,score\n2.1,14\n2.4,16\n2.8,15\n3.1,19\n3.5,21\n3.9,24",
    })
    correlation = {
        "ok": True,
        "result": {
            "test": "pearson",
            "xName": "hours",
            "yName": "score",
            "outcome": "score",
            "pairs": [(2.1, 14.0), (2.4, 16.0), (2.8, 15.0), (3.1, 19.0)],
        },
    }
    count_share = handle_analyze({"text": "18  2\n11  9"})
    cases = [
        ("group mean", group, "group means"),
        ("paired", paired, "matched before/after"),
        ("regression", regression, "scatter plot"),
        ("correlation", correlation, "scatter plot"),
        ("count-share", count_share, "count-share pie chart"),
    ]
    failures = []
    for label, engine, caption_fragment in cases:
        with tempfile.TemporaryDirectory(prefix="rowfirst-chart-test-") as output_dir:
            charts = make_charts(engine, output_dir)
            if len(charts) != 1:
                failures.append(f"{label} chart count was {len(charts)}")
                continue
            chart = charts[0]
            if caption_fragment not in chart.get("caption", ""):
                failures.append(f"{label} selected caption {chart.get('caption')!r}")
            if not Path(chart["path"]).is_file():
                failures.append(f"{label} chart file was not created")
    return failures


def reporting_regression_failures():
    """Protect DOCX generation when chart metadata points at a missing file."""
    engine = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    failures = []
    with tempfile.TemporaryDirectory(prefix="rowfirst-docx-test-") as output_dir:
        output_dir = Path(output_dir)
        missing_docx = output_dir / "Rowfirst_Results_missing_chart.docx"
        engine["charts"] = [{
            "result_index": 0,
            "path": str(output_dir / "does-not-exist.png"),
            "caption": "missing chart",
        }]
        try:
            write_docx(engine, missing_docx)
        except Exception as exc:
            failures.append(f"missing chart blocked DOCX creation: {type(exc).__name__}: {exc}")
        if not missing_docx.is_file():
            failures.append("missing chart did not produce a DOCX")

        chart_docx = output_dir / "Rowfirst_Results_with_chart.docx"
        engine["charts"] = make_charts(engine, output_dir / "charts")
        try:
            write_docx(engine, chart_docx)
        except Exception as exc:
            failures.append(f"chart embedding blocked DOCX creation: {type(exc).__name__}: {exc}")
        if not chart_docx.is_file():
            failures.append("chart embedding did not produce a DOCX")
        if chart_docx.is_file():
            with zipfile.ZipFile(chart_docx) as archive:
                if not any(name.startswith("word/media/") for name in archive.namelist()):
                    failures.append("generated chart was not embedded in DOCX")
    return failures


def telegram_regression_failures():
    """Protect the RESULTS/QA/prompt blocks and Telegram chart photo delivery."""
    class FakeChat:
        id = 42

    class FakeMessage:
        chat = FakeChat()

    class FakeBot:
        def __init__(self):
            self.replies = []
            self.photos = []

        def reply_to(self, message, text):
            self.replies.append(text)

        def send_photo(self, chat_id, image_file, caption=""):
            self.photos.append((chat_id, image_file.read(), caption))

    engine = handle_analyze({
        "text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80",
    })
    engine["qa"] = {"warnings": ["test warning"], "errors": []}
    failures = []
    with tempfile.TemporaryDirectory(prefix="rowfirst-telegram-test-") as output_dir:
        engine["charts"] = make_charts(engine, output_dir)
        bot = FakeBot()
        _send_analysis(bot, FakeMessage(), engine)
        if sum(text.startswith("RESULTS\n") for text in bot.replies) != 1:
            failures.append("RESULTS header was not sent exactly once")
        if sum(text.startswith("QA\n") for text in bot.replies) != 1:
            failures.append("QA block was not sent exactly once")
        if not any(text == "Want a Results Document (Word)? Reply YES" for text in bot.replies):
            failures.append("YES Results Document prompt was not sent")
        if len(bot.photos) != 1 or not bot.photos[0][1]:
            failures.append("chart PNG was not delivered as a Telegram photo")
    return failures


def main():
    fails = []

    A = handle_analyze({"text": "Brining: 4.22, 4.18, 4.25, 4.20\nQuick pickling: 3.71, 3.78, 3.74, 3.80"})
    r = A["result"]
    if not A["ok"]:
        fails.append("A not ok: " + A.get("error", ""))
    else:
        if not near(r["t"], 18.1396, 0.02):
            fails.append(f"A t {r['t']}")
        if r["p"] > 1e-5:
            fails.append(f"A p {r['p']}")
        if int(round(r["df"])) != 6:
            fails.append(f"A df {r['df']}")

    B = handle_analyze({"text": "Fresh: 91.2, 91.5, 90.9, 91.3\nBrining: 87.6, 88.1, 87.9, 88.3\nQuick pickling: 90.4, 90.1, 89.8, 90.6"})
    r = B["result"]
    if not B["ok"]:
        fails.append("B not ok: " + B.get("error", ""))
    else:
        if r.get("test") != "one-way anova":
            fails.append(f"B test {r.get('test')}")
        if not near(r["F"], 121.2766, 0.05):
            fails.append(f"B F {r['F']}")
        if r["p"] > 1e-5:
            fails.append(f"B p {r['p']} (must be ~3e-7, not 0.003)")
        if r.get("dfb") != 2 or r.get("dfw") != 9:
            fails.append(f"B df {r.get('dfb')},{r.get('dfw')}")

    C_result = handle_analyze({"text": "18  2\n11  9"})
    r = C_result["result"]
    if not C_result["ok"]:
        fails.append("C not ok: " + C_result.get("error", ""))
    else:
        p = r.get("p") or r.get("chi2pUncorrected")
        if p is None or not (0.01 <= p <= 0.04):
            fails.append(f"C p {p}")

    D = handle_analyze({"text": "id,before,after\n1,72,78\n2,75,79\n3,70,74\n4,80,84\n5,68,73"})
    r = D.get("result", {})
    if not D.get("ok") or r.get("test") != "paired-t":
        fails.append(f"D paired result {r}")
    else:
        if not near(r["t"], -11.5, 0.001) or not near(r["p"], 0.000326, 0.00001):
            fails.append(f"D t/p {r.get('t')}/{r.get('p')}")
        if r.get("df") != 4:
            fails.append(f"D df {r.get('df')}")

    E = handle_analyze({
        "text": (
            "method,day,score\n"
            "A,1,10\nA,1,11\nA,1,9\nA,2,12\nA,2,13\nA,2,11\n"
            "B,1,20\nB,1,21\nB,1,19\nB,2,22\nB,2,23\nB,2,21"
        )
    })
    r = E.get("result", {})
    if not E.get("ok") or r.get("test") != "two-way anova" or len(r.get("effects", [])) != 3:
        fails.append(f"E two-way result {r}")
    else:
        expected = anova_lm(
            ols("_y ~ C(_a) * C(_b)", data={
                "_a": ["A"] * 6 + ["B"] * 6,
                "_b": [1, 1, 1, 2, 2, 2] * 2,
                "_y": [10, 11, 9, 12, 13, 11, 20, 21, 19, 22, 23, 21],
            }).fit(),
            typ=2,
        )
        expected_p = [float(expected.loc[key, "PR(>F)"]) for key in ("C(_a)", "C(_b)", "C(_a):C(_b)")]
        actual_p = [effect["p"] for effect in r["effects"]]
        if any(not near(actual, target, 1e-10) for actual, target in zip(actual_p, expected_p)):
            fails.append(f"E p-values {actual_p} != {expected_p}")

    F = handle_analyze({"text": "hours,score\n2.1,14\n2.4,16\n2.8,15\n3.1,19\n3.5,21\n3.9,24"})
    r = F.get("result", {})
    expected_regression = stats.linregress([2.1, 2.4, 2.8, 3.1, 3.5, 3.9], [14, 16, 15, 19, 21, 24])
    if not F.get("ok") or r.get("test") != "simple linear regression":
        fails.append(f"F regression result {r}")
    elif not near(r["slope"], expected_regression.slope, 1e-10) or not near(r["r"], expected_regression.rvalue, 1e-10):
        fails.append(f"F slope/r {r.get('slope')}/{r.get('r')}")

    fails.extend(chart_regression_failures())
    fails.extend(reporting_regression_failures())
    fails.extend(telegram_regression_failures())

    if fails:
        print("FAIL")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("PASS A/B/C")
    print("A", A["message"])
    print("---")
    print("B", B["message"])
    print("---")
    print("C", C_result["message"])
    print("---")
    print("PASS D/E/F")
    print("D", D["message"])
    print("---")
    print("E", E["message"])
    print("---")
    print("F", F["message"])
    print("---")
    print("PASS chart selection, DOCX resilience/embedding, and Telegram reporting")


if __name__ == "__main__":
    main()
