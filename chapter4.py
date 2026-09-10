"""Results Document reports built only from the last verified engine JSON."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any


TITLE = "Results and statistical working"


def to_markdown(engine: dict[str, Any]) -> str:
    _require_engine(engine)
    results = _results(engine)
    lines = [f"# {TITLE}", "", "## 1 Study and design", "", *_preamble(engine, results), ""]

    raw_tables = _raw_tables(engine)
    if raw_tables:
        lines.extend(["## 2 Raw data table", ""])
        for title, headers, rows in raw_tables:
            if title:
                lines.extend([f"### {title}", ""])
            lines.extend(_markdown_table(headers, rows))
            lines.append("")
    else:
        lines.extend(["## 2 Raw data table", "", "Raw data were not included in the last engine JSON.", ""])

    lines.extend(["## 3 Descriptive tables", ""])
    for index, result in enumerate(results, start=1):
        if len(results) > 1:
            lines.extend([f"### {_outcome_name(result)}", ""])
        lines.extend(_markdown_table(["Treatment", "n", "Total Sum", "Mean", "SD"], _descriptive_rows(result)))
        if index < len(results):
            lines.append("")
    lines.extend(["", "## 4 Inferential table", ""])
    lines.extend(_markdown_table(
        ["Outcome", "Test", "Statistic", "df", "Exact p", "Decision"],
        _inferential_rows(results),
    ))
    lines.extend(["", "## 5 Working", ""])
    for result in results:
        lines.extend(f"- {note}" for note in _working_notes(result))
    lines.extend(["", "## 6 Interpretation", ""])
    lines.extend(_interpretations(engine))
    lines.extend(["", "## 7 Discussion", ""])
    lines.extend(_discussion(engine))
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(engine: dict[str, Any], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_markdown(engine), encoding="utf-8")
    return str(destination)


def write_docx(engine: dict[str, Any], path: str | Path) -> str:
    from docx import Document

    _require_engine(engine)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = _results(engine)
    document = Document()
    document.add_heading(TITLE, level=1)
    document.add_paragraph("Statistical analysis and working built from the verified engine JSON.")
    document.add_heading("1 Study and design", level=2)
    for paragraph in _preamble(engine, results):
        document.add_paragraph(paragraph)

    document.add_heading("2 Raw data table", level=2)
    raw_tables = _raw_tables(engine)
    if raw_tables:
        for title, headers, rows in raw_tables:
            if title:
                document.add_heading(title, level=3)
            _add_docx_table(document, headers, rows)
    else:
        document.add_paragraph("Raw data were not included in the last engine JSON.")

    document.add_heading("3 Descriptive tables", level=2)
    for index, result in enumerate(results):
        if len(results) > 1:
            document.add_heading(_outcome_name(result), level=3)
        _add_docx_table(document, ["Treatment", "n", "Total Sum", "Mean", "SD"], _descriptive_rows(result))
        _embed_chart(document, engine, index)

    document.add_heading("4 Inferential table", level=2)
    _add_docx_table(
        document,
        ["Outcome", "Test", "Statistic", "df", "Exact p", "Decision"],
        _inferential_rows(results),
    )

    document.add_heading("5 Working", level=2)
    for result in results:
        for note in _working_notes(result):
            document.add_paragraph(note, style="List Bullet")

    document.add_heading("6 Interpretation", level=2)
    for paragraph in _interpretations(engine):
        document.add_paragraph(paragraph)
    document.add_heading("7 Discussion", level=2)
    for paragraph in _discussion(engine):
        document.add_paragraph(paragraph)
    document.save(destination)
    return str(destination)


def write_pdf(engine: dict[str, Any], path: str | Path) -> str:
    """Render the same engine-JSON report as a PDF."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    _require_engine(engine)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = _results(engine)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RowfirstBody", parent=styles["BodyText"], alignment=TA_LEFT, leading=14))
    styles.add(ParagraphStyle(name="RowfirstSmall", parent=styles["BodyText"], alignment=TA_LEFT, fontSize=8, leading=10))
    story = [
        Paragraph(_escape(TITLE), styles["Title"]),
        PageBreak(),
        Paragraph("1 Study and design", styles["Heading2"]),
    ]
    story.extend(Paragraph(_escape(paragraph), styles["RowfirstBody"]) for paragraph in _preamble(engine, results))
    story.extend([Spacer(1, 0.12 * inch), Paragraph("2 Raw data table", styles["Heading2"])])
    raw_tables = _raw_tables(engine)
    if raw_tables:
        for title, headers, rows in raw_tables:
            if title:
                story.append(Paragraph(_escape(title), styles["Heading4"]))
            story.append(_pdf_table(headers, rows, styles))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("Raw data were not included in the last engine JSON.", styles["RowfirstBody"]))

    story.append(Paragraph("3 Descriptive tables", styles["Heading2"]))
    for result in results:
        if len(results) > 1:
            story.append(Paragraph(_escape(_outcome_name(result)), styles["Heading3"]))
        story.append(_pdf_table(["Treatment", "n", "Total Sum", "Mean", "SD"], _descriptive_rows(result), styles))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("4 Inferential table", styles["Heading2"]))
    story.append(_pdf_table(
        ["Outcome", "Test", "Statistic", "df", "Exact p", "Decision"],
        _inferential_rows(results),
        styles,
    ))

    story.extend([Spacer(1, 0.18 * inch), Paragraph("5 Working", styles["Heading2"])])
    for result in results:
        for note in _working_notes(result):
            story.append(Paragraph(_escape(note), styles["RowfirstBody"]))
    story.extend([Spacer(1, 0.18 * inch), Paragraph("6 Interpretation", styles["Heading2"])])
    story.extend(Paragraph(_escape(paragraph), styles["RowfirstBody"]) for paragraph in _interpretations(engine))
    story.extend([Spacer(1, 0.18 * inch), Paragraph("7 Discussion", styles["Heading2"])])
    story.extend(Paragraph(_escape(paragraph), styles["RowfirstBody"]) for paragraph in _discussion(engine))

    SimpleDocTemplate(
        str(destination),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    ).build(story)
    return str(destination)


def _preamble(engine: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    lines = []
    topic = engine.get("topic")
    if topic:
        lines.append(f"Topic: {topic}.")
    lines.extend([
        f"The analysis used a {_design(results)} design.",
        _sample_description(results),
        "All decisions were made at α = .05.",
    ])
    lines.extend(_hypotheses(engine, results))
    return lines


def _hypotheses(engine: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    supplied = engine.get("hypotheses") or []
    if supplied:
        lines = []
        for item in supplied:
            if isinstance(item, dict):
                label = str(item.get("label") or "H0").strip()
                text = str(item.get("text") or "").strip()
            else:
                label, text = "H0", str(item).strip()
            if text:
                lines.append(f"{label}: {text}")
        if lines:
            return lines
    hypothesis = str(engine.get("ho") or "").strip()
    if hypothesis:
        return [f"H0: {hypothesis.removeprefix('H0:').removeprefix('Ho:').strip()}"]
    return [
        f"H0: no difference between groups on {_outcome_name(result)}."
        for result in results
    ]


def _subject(engine: dict[str, Any], results: list[dict[str, Any]]) -> str:
    topic = engine.get("topic")
    if topic:
        return str(topic)
    names = []
    for result in results:
        if result.get("parameter"):
            names.append(str(result["parameter"]))
        elif result.get("test") == "two-way anova":
            names.append(str(result.get("outcome", "the measured outcome")))
        elif result.get("test") == "simple linear regression":
            names.append(f"{result.get('predictor', 'predictor')} and {result.get('outcome', 'outcome')}")
        elif result.get("test") in {"pearson", "spearman"}:
            names.append(f"{result.get('xName', 'X')} and {result.get('yName', 'Y')}")
        elif result.get("test") in {"student-t", "welch-t", "one-way anova"}:
            names.append("the treatment groups")
        elif result.get("test") == "paired-t":
            names.append(f"{result.get('before', {}).get('name', 'before')} and {result.get('after', {}).get('name', 'after')}")
        else:
            names.append("the contingency table")
    return ", ".join(dict.fromkeys(names)) or "the submitted data"


def _design(results: list[dict[str, Any]]) -> str:
    labels = []
    for result in results:
        label = {
            "student-t": "independent-samples comparison",
            "welch-t": "independent-samples comparison",
            "paired-t": "paired comparison",
            "one-way anova": "one-way ANOVA",
            "two-way anova": "two-way ANOVA",
            "simple linear regression": "simple linear regression",
            "pearson": "correlation",
            "spearman": "correlation",
            "fisher-exact": "contingency-table analysis",
            "chi-square": "contingency-table analysis",
        }.get(result.get("test"), str(result.get("test") or "the selected statistical test"))
        labels.append(label)
    return ", ".join(dict.fromkeys(labels)) or "the selected statistical test"


def _sample_description(results: list[dict[str, Any]]) -> str:
    group_sizes: list[tuple[str, Any]] = []
    other_sizes: list[str] = []
    for result in results:
        test = result.get("test")
        if test in {"student-t", "welch-t"}:
            group_sizes.extend(
                (str(group["name"]), group["n"])
                for group in (result["group1"], result["group2"])
            )
        elif test == "one-way anova":
            group_sizes.extend(
                (str(group["name"]), group["n"]) for group in result.get("groups", [])
            )
        elif test == "paired-t":
            other_sizes.append(f"n = {result['nPairs']} matched pairs")
        elif "n" in result:
            other_sizes.append(f"n = {result['n']}")
    unique_groups = list(dict.fromkeys(group_sizes))
    if unique_groups:
        sizes = list(dict.fromkeys(size for _, size in unique_groups))
        if len(sizes) == 1:
            names = ", ".join(name for name, _ in unique_groups)
            return f"n = {sizes[0]} per group: {names}."
        return "Sample sizes: " + "; ".join(
            f"n = {size} for {name}" for name, size in unique_groups
        ) + "."
    if other_sizes:
        return "Sample sizes: " + "; ".join(dict.fromkeys(other_sizes)) + "."
    return "Sample size: not reported by the engine."


def _raw_tables(engine: dict[str, Any]) -> list[tuple[str, list[str], list[tuple[str, ...]]]]:
    ingested = engine.get("ingested") or {}
    fmt = ingested.get("format")
    if fmt in {"long-multi", "factor-multi", "wide-multi"}:
        tables = []
        for outcome in ingested.get("outcomes", []):
            parameter = str(outcome.get("parameter") or "Value")
            rows = [
                (str(group["name"]), _raw_value(value))
                for group in outcome.get("groups", [])
                for value in group.get("values", [])
            ]
            tables.append((parameter, ["Treatment", parameter], rows))
        return tables
    if fmt == "labelled":
        rows = [
            (str(group["name"]), _raw_value(value))
            for group in ingested.get("groups", [])
            for value in group.get("values", [])
        ]
        return [("", ["Treatment", "Value"], rows)]
    if fmt == "paired":
        headers = [str(ingested.get("id", "ID")), str(ingested.get("before", "Before")), str(ingested.get("after", "After"))]
        rows = [
            (str(pair["id"]), _raw_value(pair["before"]), _raw_value(pair["after"]))
            for pair in ingested.get("pairs", [])
        ]
        return [("", headers, rows)]
    if fmt == "two-way":
        headers = [str(ingested.get("factorA", "Factor A")), str(ingested.get("factorB", "Factor B")), str(ingested.get("outcome", "Outcome"))]
        rows = [
            tuple(_raw_value(row.get(header, "")) for header in headers)
            for row in ingested.get("rows", [])
        ]
        return [("", headers, rows)]
    if fmt == "wide":
        groups = ingested.get("groups", [])
        width = max((len(group.get("values", [])) for group in groups), default=0)
        rows = []
        for index in range(width):
            rows.append(tuple([str(index + 1)] + [
                _raw_value(group["values"][index]) if index < len(group.get("values", [])) else ""
                for group in groups
            ]))
        return [("", ["Observation"] + [str(group["name"]) for group in groups], rows)]
    if fmt == "contingency":
        matrix = ingested.get("matrix", [])
        width = max((len(row) for row in matrix), default=0)
        rows = [
            tuple([str(index + 1)] + [_raw_value(value) for value in row] + [""] * (width - len(row)))
            for index, row in enumerate(matrix)
        ]
        return [("", ["Row"] + [f"Column {index + 1}" for index in range(width)], rows)]
    return []


def _raw_value(value: Any) -> str:
    return "" if value is None else str(value)


def _descriptive_rows(result: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        return [_group_row(result["group1"]), _group_row(result["group2"])]
    if test == "one-way anova":
        return [_group_row(group) for group in result.get("groups", [])]
    if test == "paired-t":
        return [_group_row(result["before"]), _group_row(result["after"])]
    if test == "two-way anova":
        return [("Factor combinations", str(result.get("n", "not reported")), "not reported", "not reported", "not reported")]
    if test in {"simple linear regression", "pearson", "spearman"}:
        label = f"{result.get('predictor', result.get('xName', 'X'))} / {result.get('outcome', result.get('yName', 'Y'))}"
        return [(label, str(result.get("n", "not reported")), "not reported", "not reported", "not reported")]
    return [("Count table", "not reported", "not applicable", "not applicable", "not applicable")]


def _group_row(group: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(group.get("name", "Group")),
        str(group.get("n", "not reported")),
        _fmt_sum(group.get("totalSum")),
        _fmt_num(group.get("mean")),
        _fmt_num(group.get("sd")),
    )


def _inferential_rows(results: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    for result in results:
        name = _outcome_name(result)
        test = result.get("test")
        if test in {"student-t", "welch-t", "paired-t"}:
            rows.append((
                name,
                "Paired t-test" if test == "paired-t" else ("Welch t-test" if test == "welch-t" else "Independent t-test"),
                f"t = {result['t']:.12g}",
                _df(result["df"]),
                _fmt_p(result["p"]),
                _decision(result.get("isSignificant", False)),
            ))
        elif test == "one-way anova":
            rows.append((name, "One-way ANOVA", f"F = {result['F']:.12g}", f"{result['dfb']}, {result['dfw']}", _fmt_p(result["p"]), _decision(result.get("isSignificant", False))))
        elif test == "two-way anova":
            effects = result.get("effects", [])
            rows.append((
                name,
                "Two-way ANOVA",
                "; ".join(f"{effect['effect']}: F = {effect['F']:.12g}" for effect in effects) or "not reported",
                "; ".join(f"{effect['df']:.0f}, {result['residualDf']:.0f}" for effect in effects) or "—",
                "; ".join(_fmt_p(effect.get("p")) for effect in effects) or "not reported",
                "; ".join(
                    f"{effect['effect']}: {_decision(effect.get('isSignificant', False))}"
                    for effect in effects
                ) or "not reported",
            ))
        elif test == "simple linear regression":
            rows.append((name, "Simple linear regression", f"r = {result['r']:.12g}", "—", _fmt_p(result["p"]), _decision(result.get("isSignificant", False))))
        elif test in {"pearson", "spearman"}:
            rows.append((name, f"{test.title()} correlation", f"r = {result['r']:.12g}", str(result.get("n", "—")), _fmt_p(result["p"]), _decision(result.get("isSignificant", False))))
        elif test == "fisher-exact":
            rows.append((name, "Fisher exact test", "Exact test", "—", _fmt_p(result["p"]), _decision(result.get("isSignificant", False))))
        else:
            rows.append((name, "Chi-square test", f"χ² = {result['chi2']:.12g}", str(result.get("df", "—")), _fmt_p(result["p"]), _decision(result.get("isSignificant", False))))
    return rows


def _working_notes(result: dict[str, Any]) -> list[str]:
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        return [
            f"Formula: {'Welch' if test == 'welch-t' else 'independent-samples'} t-test; n={result['group1']['n']} and n={result['group2']['n']}.",
            f"{result['group1']['name']}: mean={result['group1']['mean']:.3f}, SD={result['group1']['sd']:.3f}; {result['group2']['name']}: mean={result['group2']['mean']:.3f}, SD={result['group2']['sd']:.3f}.",
            f"t={result['t']:.12g}, df={_df(result['df'])}, exact p={_fmt_p(result['p'])}.",
        ]
    if test == "paired-t":
        return [
            f"Formula: paired t-test on matched differences; n={result['nPairs']}.",
            f"{result['before']['name']}: mean={result['before']['mean']:.3f}, SD={result['before']['sd']:.3f}; {result['after']['name']}: mean={result['after']['mean']:.3f}, SD={result['after']['sd']:.3f}.",
            f"t={result['t']:.12g}, df={_df(result['df'])}, exact p={_fmt_p(result['p'])}.",
        ]
    if test == "one-way anova":
        group_text = "; ".join(f"{g['name']}: n={g['n']}, mean={g['mean']:.3f}, SD={g['sd']:.3f}" for g in result.get("groups", []))
        notes = [
            f"Formula: one-way ANOVA; {group_text}.",
            f"F={result['F']:.12g}, df={result['dfb']}, {result['dfw']}, exact p={_fmt_p(result['p'])}.",
        ]
        effect_size = result.get("effectSize", {})
        if effect_size:
            notes.append(
                f"Effect size: eta-squared={effect_size.get('etaSquared', 0.0):.12g}, "
                f"omega-squared={effect_size.get('omegaSquared', 0.0):.12g}."
            )
        assumptions = result.get("assumptions", {})
        shapiro = "; ".join(
            f"{item['group']}: p={_fmt_p(item['p'])}"
            for item in assumptions.get("shapiroWilk", [])
            if item.get("p") is not None
        )
        if shapiro:
            notes.append(f"Shapiro-Wilk normality checks: {shapiro}.")
        levene = assumptions.get("levene", {})
        if levene.get("p") is not None:
            notes.append(
                f"Levene variance homogeneity check: W={levene['W']:.12g}, "
                f"p={_fmt_p(levene['p'])}."
            )
        post_hoc = result.get("postHoc")
        if post_hoc:
            comparisons = "; ".join(
                f"{item['group1']} vs {item['group2']}: p_adj={_fmt_p(item['pAdjusted'])}"
                f"{' (significant)' if item['reject'] else ''}"
                for item in post_hoc.get("comparisons", [])
            )
            if comparisons:
                notes.append(f"Tukey HSD post-hoc pairwise comparisons: {comparisons}.")
        return notes
    if test == "two-way anova":
        return [
            f"Formula: two-way ANOVA with {result['factorA']} × {result['factorB']}; n={result['n']}.",
            *[f"{effect['effect']}: F={effect['F']:.12g}, df={effect['df']:.0f}, {result['residualDf']:.0f}, exact p={_fmt_p(effect['p'])}." for effect in result.get("effects", [])],
        ]
    if test == "simple linear regression":
        return [
            f"Formula: simple linear regression; n={result['n']}; predictor={result['predictor']}; outcome={result['outcome']}.",
            f"slope={result['slope']:.12g}, intercept={result['intercept']:.12g}, r={result['r']:.12g}, exact p={_fmt_p(result['p'])}.",
        ]
    if test in {"pearson", "spearman"}:
        return [f"Formula: {test} correlation; n={result['n']}; r={result['r']:.12g}; exact p={_fmt_p(result['p'])}."]
    if test == "fisher-exact":
        return [f"Formula: Fisher exact test for a 2×2 count table; exact p={_fmt_p(result['p'])}."]
    return [f"Formula: chi-square test; χ²={result['chi2']:.12g}, df={result['df']}, exact p={_fmt_p(result['p'])}."]


def _interpretations(engine: dict[str, Any]) -> list[str]:
    results = _results(engine)
    if not results:
        return ["The engine did not provide an interpretation."]
    clauses = [_interpretation_clause(result) for result in results]
    return [" ".join(clauses) + " " + _study_significance_sentence(results)]


def _discussion(engine: dict[str, Any]) -> list[str]:
    results = _results(engine)
    if not results:
        return ["No discussion was produced because the engine returned no result."]
    topic = str(engine.get("topic") or "").strip()
    synthesis = _synthesis_paragraph(results, topic)
    return [synthesis, _discussion_limits(results)]


def _interpretation_clause(result: dict[str, Any]) -> str:
    label = _outcome_name(result)
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        groups = [result["group1"], result["group2"]]
        higher, lower = sorted(groups, key=lambda group: group["mean"], reverse=True)
        return (
            f"{label}: {higher['name']} was higher ({higher['mean']:.3f}) "
            f"than {lower['name']} ({lower['mean']:.3f}), t({_df(result['df'])}) = "
            f"{result['t']:.4f}, p = {_fmt_p(result['p'])}."
        )
    if test == "one-way anova":
        groups = result.get("groups", [])
        if not groups:
            return f"{label}: group means were not reported; F({result['dfb']}, {result['dfw']}) = {result['F']:.4f}, p = {_fmt_p(result['p'])}."
        higher = max(groups, key=lambda group: group["mean"])
        lower = min(groups, key=lambda group: group["mean"])
        return (
            f"{label}: {higher['name']} was highest ({higher['mean']:.3f}) and "
            f"{lower['name']} was lowest ({lower['mean']:.3f}), F({result['dfb']}, "
            f"{result['dfw']}) = {result['F']:.4f}, p = {_fmt_p(result['p'])}."
        )
    if test == "paired-t":
        before, after = result["before"], result["after"]
        direction = "higher" if after["mean"] > before["mean"] else "lower" if after["mean"] < before["mean"] else "the same as"
        return (
            f"{label}: {after['name']} was {direction} {before['name']} "
            f"({after['mean']:.3f} versus {before['mean']:.3f}), t({_df(result['df'])}) = "
            f"{result['t']:.4f}, p = {_fmt_p(result['p'])}."
        )
    if test == "two-way anova":
        effect_text = "; ".join(
            f"{effect['effect']}: F({effect['df']:.0f}, {result['residualDf']:.0f}) = "
            f"{effect['F']:.4f}, p = {_fmt_p(effect['p'])}"
            for effect in result.get("effects", [])
        )
        return f"{label}: {effect_text or 'effects were not reported'}."
    if test == "simple linear regression":
        direction = "higher" if result["slope"] >= 0 else "lower"
        return (
            f"{label}: higher {result['predictor']} values went with {direction} "
            f"{result['outcome']} values, r = {result['r']:.4f}, p = {_fmt_p(result['p'])}."
        )
    if test in {"pearson", "spearman"}:
        direction = "together" if result["r"] >= 0 else "in opposite directions"
        return f"{label}: the two columns moved {direction}, r = {result['r']:.4f}, p = {_fmt_p(result['p'])}."
    if test == "fisher-exact":
        return f"{label}: the two-by-two categories showed {'an association' if result.get('isSignificant') else 'no clear association'}, Fisher p = {_fmt_p(result['p'])}."
    if test == "chi-square":
        return f"{label}: the observed counts {'differed from' if result.get('isSignificant') else 'did not clearly differ from'} chance expectations, χ²({result['df']}) = {result['chi2']:.4f}, p = {_fmt_p(result['p'])}."
    return f"{label}: the direction and engine statistic were not reported."


def _study_significance_sentence(results: list[dict[str, Any]]) -> str:
    significant: list[str] = []
    not_significant: list[str] = []
    for result in results:
        label = _outcome_name(result)
        if result.get("test") == "two-way anova":
            for effect in result.get("effects", []):
                target = f"{label} ({effect['effect']})"
                (significant if effect.get("isSignificant") else not_significant).append(target)
        else:
            (significant if result.get("isSignificant") else not_significant).append(label)
    if significant and not_significant:
        return f"At 5%, significant evidence was found for {_join_items(significant)}, but not for {_join_items(not_significant)}."
    if significant:
        return f"At 5%, significant evidence was found for {_join_items(significant)}."
    return f"At 5%, none of the tested outcomes showed significant evidence."


def _discussion_significance_sentence(results: list[dict[str, Any]]) -> str:
    significant: list[str] = []
    not_significant: list[str] = []
    for result in results:
        if result.get("test") == "two-way anova":
            for effect in result.get("effects", []):
                target = f"the {effect['effect']} effect on {_outcome_name(result)}"
                (significant if effect.get("isSignificant") else not_significant).append(target)
        else:
            target = _discussion_evidence_target(result)
            (significant if result.get("isSignificant") else not_significant).append(target)
    if significant and not_significant:
        return (
            f"At the 5% level, the data supported {_join_items(significant)}, "
            f"but the evidence was not clear for {_join_items(not_significant)}."
        )
    if significant:
        return f"At the 5% level, the data supported {_join_items(significant)}."
    return (
        "The observed pattern may be meaningful, but the tests did not provide clear "
        "statistical evidence at the 5% level."
    )


def _discussion_evidence_target(result: dict[str, Any]) -> str:
    test = result.get("test")
    if test == "simple linear regression":
        return (
            f"the relationship between {result.get('predictor', 'the predictor')} "
            f"and {result.get('outcome', 'the outcome')}"
        )
    if test in {"pearson", "spearman"}:
        return (
            f"the relationship between {result.get('xName', 'the first column')} "
            f"and {result.get('yName', 'the second column')}"
        )
    if test in {"fisher-exact", "chi-square"}:
        return "the association in the count categories"
    if test == "paired-t":
        return (
            f"the difference between {result['after']['name']} "
            f"and {result['before']['name']}"
        )
    return f"the difference in {_outcome_name(result)}"


def _synthesis_paragraph(results: list[dict[str, Any]], topic: str) -> str:
    if results and all(result.get("test") == "paired-t" for result in results):
        story = "; ".join(
            _paired_synthesis_fragment(result) for result in results
        ) + "."
        context = f"Taken together in {topic}, " if topic else "Taken together, "
        return context + story + " " + _discussion_significance_sentence(results)
    group_directions = [_group_direction(result) for result in results]
    group_directions = [item for item in group_directions if item]
    other_results = [
        result
        for result in results
        if _group_direction(result) is None
    ]
    if group_directions:
        high_groups: dict[str, list[str]] = {}
        low_groups: dict[str, list[str]] = {}
        for label, higher, lower in group_directions:
            high_groups.setdefault(higher, []).append(label)
            low_groups.setdefault(lower, []).append(label)
        max_count = max(len(labels) for labels in high_groups.values())
        standouts = [name for name, labels in high_groups.items() if len(labels) == max_count]
        if len(standouts) == 1:
            high_text = _direction_story(
                standouts[0],
                high_groups[standouts[0]],
                "higher",
            )
        else:
            high_text = "; ".join(
                _direction_story(name, high_groups[name], "higher")
                for name in standouts
            )
        lower_text = "; ".join(
            _direction_story(name, labels, "lower")
            for name, labels in low_groups.items()
        )
        story = f"{high_text}; {lower_text}."
        if other_results:
            story = story[:-1] + "; " + "; ".join(
                _synthesis_fragment(result) for result in other_results
            ) + "."
    else:
        story = "; ".join(_synthesis_fragment(result) for result in results) + "."
    context = f"Taken together in {topic}, " if topic else "Taken together, "
    return context + story + " " + _discussion_significance_sentence(results)


def _direction_story(name: str, labels: list[str], direction: str) -> str:
    return f"{name} " + _join_items([
        _direction_phrase(label, direction) for label in labels
    ])


def _direction_phrase(label: str, direction: str) -> str:
    clean = str(label).strip()
    lower = clean.lower()
    if re.search(r"\bpH\b", clean, flags=re.I):
        return (
            f"had a {'higher' if direction == 'higher' else 'lower'} pH "
            f"({'less acidic' if direction == 'higher' else 'more acidic'})"
        )
    if re.search(r"\bmoisture\b|\bwater(?:\s+content)?\b|water_content", lower):
        return "had higher moisture" if direction == "higher" else "was drier"
    if re.search(r"\btime\b|\bduration\b", lower):
        return f"had a {'longer' if direction == 'higher' else 'shorter'} {clean}"
    if lower == "score" or lower.endswith(" score"):
        return f"had a {direction} {clean}"
    return f"had {direction} values for {clean}"


def _paired_synthesis_fragment(result: dict[str, Any]) -> str:
    before, after = result["before"], result["after"]
    if after["mean"] > before["mean"]:
        direction = "higher than"
    elif after["mean"] < before["mean"]:
        direction = "lower than"
    else:
        direction = "the same as"
    suffix = f" on {_outcome_name(result)}" if result.get("parameter") else ""
    return f"{after['name']} was {direction} {before['name']}{suffix}"


def _discussion_limits(results: list[dict[str, Any]]) -> str:
    clauses = []
    if _has_sample_size(results):
        clauses.append("Small n limits how widely this pattern can be generalized")
    if any(result.get("test") in {"one-way anova", "two-way anova"} for result in results):
        clauses.append("ANOVA does not establish that every pair of groups differs")
    clauses.append("No causation is established")
    clauses.append(_discussion_outcome_limit(results))
    return "Limits: " + ". ".join(clauses) + "."


def _discussion_outcome_limit(results: list[dict[str, Any]]) -> str:
    labels = [_outcome_name(result) for result in results]
    label_text = _join_items(labels)
    lower_labels = " ".join(labels).lower()
    if re.search(r"\bdwell\b|\btime\b|\bduration\b|\bseconds?\b|\bminutes?\b", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe a longer or shorter duration only; "
            "conclusions about process performance, quality, or safety require additional domain evidence"
        )
    if re.search(r"\bph\b", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe acidity only; "
            "quality or safety conclusions require additional domain evidence"
        )
    if re.search(r"\bmoisture\b|\bwater(?:\s+content)?\b|water_content", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe moisture only; "
            "quality or safety conclusions require additional domain evidence"
        )
    return (
        f"Higher or lower values on {label_text} describe the measured response only; "
        "practical conclusions require additional domain evidence"
    )


def _has_sample_size(results: list[dict[str, Any]]) -> bool:
    return _count_total(results) is not None or any(
        result.get("n") is not None
        or result.get("nPairs") is not None
        or any(group.get("n") is not None for group in result.get("groups", []))
        or any(
            isinstance(result.get(key), dict) and result[key].get("n") is not None
            for key in ("before", "after", "group1", "group2")
        )
        for result in results
    )


def _count_total(results: list[dict[str, Any]]) -> int | None:
    totals = []
    for result in results:
        if result.get("test") not in {"fisher-exact", "chi-square"}:
            continue
        matrix = result.get("matrix")
        if not isinstance(matrix, list):
            continue
        values = [
            value
            for row in matrix
            if isinstance(row, list)
            for value in row
            if isinstance(value, (int, float))
        ]
        if values:
            totals.append(int(sum(values)))
    return sum(totals) if totals else None


def _has_moisture_outcome(results: list[dict[str, Any]]) -> bool:
    fields = []
    for result in results:
        for key in ("parameter", "outcome", "xName", "yName"):
            if result.get(key):
                fields.append(str(result[key]))
        for key in ("before", "after"):
            if isinstance(result.get(key), dict) and result[key].get("name"):
                fields.append(str(result[key]["name"]))
    text = " ".join(fields).lower()
    return bool(re.search(r"\bmoisture\b|\bwater(?:\s+content)?\b|water_content", text))


def _group_direction(result: dict[str, Any]) -> tuple[str, str, str] | None:
    label = _outcome_name(result)
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        higher, lower = sorted((result["group1"], result["group2"]), key=lambda group: group["mean"], reverse=True)
        return label, str(higher["name"]), str(lower["name"])
    if test == "one-way anova" and result.get("groups"):
        higher = max(result["groups"], key=lambda group: group["mean"])
        lower = min(result["groups"], key=lambda group: group["mean"])
        return label, str(higher["name"]), str(lower["name"])
    if test == "paired-t":
        before, after = result["before"], result["after"]
        return label, str(after["name"]), str(before["name"])
    return None


def _synthesis_fragment(result: dict[str, Any]) -> str:
    label = _outcome_name(result)
    test = result.get("test")
    if test == "two-way anova":
        effects = ", ".join(str(effect["effect"]) for effect in result.get("effects", []))
        return f"{label} effects across {effects or 'the tested factors'}"
    if test == "simple linear regression":
        return f"{label} rose with {result['predictor']}" if result["slope"] >= 0 else f"{label} fell as {result['predictor']} rose"
    if test in {"pearson", "spearman"}:
        return f"{label} moved with its paired column" if result["r"] >= 0 else f"{label} moved against its paired column"
    return f"{label} changed across the submitted data"


def _join_items(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _outcome_name(result: dict[str, Any]) -> str:
    if result.get("parameter") or result.get("outcome") or result.get("outcomeName"):
        return str(result.get("parameter") or result.get("outcome") or result.get("outcomeName"))
    if result.get("test") == "paired-t":
        return f"{result.get('before', {}).get('name', 'before')} vs {result.get('after', {}).get('name', 'after')}"
    if result.get("test") in {"fisher-exact", "chi-square"}:
        return "count categories"
    return "Measured outcome"


def _decision(significant: bool) -> str:
    return "Reject H0" if significant else "Fail to reject H0"


def _fmt_num(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "not reported"


def _fmt_sum(value: Any) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "not reported"


def _fmt_p(value: Any) -> str:
    return f"{float(value):.12g}" if isinstance(value, (int, float)) else "not reported"


def _df(value: Any) -> str:
    number = float(value)
    return str(int(number)) if abs(number - round(number)) < 1e-6 else f"{number:.12g}"


def _markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _add_docx_table(document: Any, headers: list[str], rows: list[tuple[str, ...]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, heading in zip(table.rows[0].cells, headers):
        cell.text = str(heading)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)


def _embed_chart(document: Any, engine: dict[str, Any], result_index: int) -> None:
    """Embed a matching chart when available; a missing PNG never blocks DOCX output."""
    chart = next(
        (
            item for item in (engine.get("charts") or [])
            if item.get("result_index") == result_index
        ),
        None,
    )
    if not chart:
        return
    chart_path = Path(str(chart.get("path", "")))
    if not chart_path.is_file():
        return
    try:
        from docx.shared import Inches

        document.add_paragraph(str(chart.get("caption") or "Chart"))
        document.add_picture(str(chart_path), width=Inches(6.2))
    except Exception:
        return


def _pdf_table(headers: list[str], rows: list[tuple[str, ...]], styles: Any) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    data = [[Paragraph(_escape(value), styles["RowfirstSmall"]) for value in headers]]
    data.extend([Paragraph(_escape(str(value)), styles["RowfirstSmall"]) for value in row] for row in rows)
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _results(engine: dict[str, Any]) -> list[dict[str, Any]]:
    results = engine.get("results") or [engine.get("result")]
    return [result for result in results if result]


def _require_engine(engine: dict[str, Any]) -> None:
    if not engine or not engine.get("ok", True):
        raise ValueError("A successful engine JSON result is required.")
    if not _results(engine):
        raise ValueError("Engine JSON contains no result.")