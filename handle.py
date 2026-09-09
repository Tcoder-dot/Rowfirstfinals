"""Public analysis API: handle_analyze({text}) -> verified JSON and RESULTS text."""
from __future__ import annotations

import json
import re
from typing import Any

from ingest import ingest_text
from qa import quality_check
from stats_engine import (
    analyze_groups,
    chi_or_fisher,
    correlation,
    linear_regression,
    paired_ttest,
    two_way_anova,
)


UNSUPPORTED = (
    "I can analyse supported descriptive statistics, tests, correlations, and one-predictor "
    "linear regression. Forecasting, multiple regression, GLM, logistic regression, mixed models, "
    "and survival analysis are not supported."
)


def handle_analyze(req: dict) -> dict:
    try:
        request = req or {}
        text = request.get("text") or ""
        if _is_unsupported(text):
            return {"ok": False, "unsupported": True, "error": UNSUPPORTED}
        ingested = ingest_text(text)
        engine = analyze_ingested(
            ingested,
            mode=_requested_mode(text, request),
            outcome_name=_requested_outcome(request),
        )
        engine["qa"] = quality_check(ingested)
        engine["breakdown"] = build_breakdown(engine)
        if request.get("study") or request.get("topic"):
            engine["topic"] = request.get("topic") or request.get("study")
        if request.get("hypotheses"):
            engine["hypotheses"] = request["hypotheses"]
        elif request.get("ho"):
            engine["hypotheses"] = [{"label": "H0", "text": request["ho"]}]
        return engine
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def analyze_ingested(
    ingested: dict,
    mode: str | None = None,
    outcome_name: str | None = None,
) -> dict:
    """Run the verified engine over already-ingested local data."""
    fmt = ingested.get("format")
    if fmt == "needs-clarification":
        return {
            "ok": False,
            "needsClarification": True,
            "error": ingested.get("question", "Please identify the intended design before analysis."),
            "question": ingested.get("question"),
            "ingested": ingested,
        }
    if fmt == "contingency":
        result = chi_or_fisher(ingested["matrix"])
        return _success(ingested, [result])
    if fmt == "paired":
        pairs = ingested["pairs"]
        result = paired_ttest(
            [pair["before"] for pair in pairs],
            [pair["after"] for pair in pairs],
            [str(pair["id"]) for pair in pairs],
        )
        result["before"]["name"] = ingested["before"]
        result["after"]["name"] = ingested["after"]
        return _success(ingested, [result])
    if fmt == "two-way":
        result = two_way_anova(ingested["rows"], ingested["factorA"], ingested["factorB"], ingested["outcome"])
        if outcome_name:
            result["outcome"] = outcome_name
        return _success(ingested, [result])

    outcomes = ingested.get("outcomes")
    if outcomes:
        runs = []
        for item in outcomes:
            result = analyze_groups(item["groups"], outcome=item["parameter"])
            result["parameter"] = item["parameter"]
            runs.append(result)
        return _success(ingested, runs)

    if fmt == "wide" and len(ingested.get("groups", [])) == 2:
        first, second = ingested["groups"]
        if len(first["values"]) != len(second["values"]):
            raise ValueError("The two numeric columns must have the same number of observations.")
        if mode == "correlation":
            result = correlation(first["values"], second["values"])
        else:
            result = linear_regression(
                first["values"],
                second["values"],
                first["name"],
                outcome_name or second["name"],
            )
        result["xName"] = first["name"]
        result["yName"] = outcome_name or second["name"]
        result["pairs"] = list(zip(first["values"], second["values"]))
        return _success(ingested, [result])
    result = analyze_groups(ingested["groups"], outcome=outcome_name)
    if outcome_name and not result.get("parameter"):
        result["parameter"] = outcome_name
    return _success(ingested, [result])


def _success(ingested: dict, results: list[dict[str, Any]]) -> dict:
    return {
        "ok": True,
        "ingested": ingested,
        "results": results,
        "result": results[0],
        "message": "\n\n".join(format_result(result) for result in results),
    }


def format_result(r: dict) -> str:
    param = r.get("parameter")
    title = f"{param}\n" if param else ""
    test = r.get("test")
    if test in ("student-t", "welch-t"):
        a, b = r["group1"], r["group2"]
        method = "Welch t-test" if test == "welch-t" else "Independent-samples t-test"
        return (
            f"{title}{method}\n"
            f"{a['name']}: mean={a['mean']:.4f}, SD={a['sd']:.4f}, n={a['n']}\n"
            f"{b['name']}: mean={b['mean']:.4f}, SD={b['sd']:.4f}, n={b['n']}\n"
            f"t({_df(r['df'])}) = {r['t']:.4f}, p = {_p(r['p'])}\n"
            f"Cohen's d = {r['cohensD']:.4f}\n"
            f"{'Significant at α = .05.' if r['isSignificant'] else 'Not significant at α = .05.'}"
        )
    if test == "paired-t":
        return (
            f"{title}Paired t-test\n"
            f"{r['before']['name']}: mean={r['before']['mean']:.4f}, SD={r['before']['sd']:.4f}, n={r['before']['n']}\n"
            f"{r['after']['name']}: mean={r['after']['mean']:.4f}, SD={r['after']['sd']:.4f}, n={r['after']['n']}\n"
            f"mean difference (after − before) = {r['meanDifference']:.4f}\n"
            f"t({_df(r['df'])}) = {r['t']:.4f}, p = {_p(r['p'])}\n"
            f"Cohen's dz = {r['cohensDz']:.4f}\n"
            f"{'Significant at α = .05.' if r['isSignificant'] else 'Not significant at α = .05.'}"
        )
    if test == "one-way anova":
        lines = [f"{title}One-way ANOVA"]
        lines.extend(f"{g['name']}: mean={g['mean']:.4f}, SD={g['sd']:.4f}, n={g['n']}" for g in r["groups"])
        lines.append(f"F({r['dfb']}, {r['dfw']}) = {r['F']:.4f}, p = {_p(r['p'])}")
        effect_size = r.get("effectSize", {})
        if effect_size:
            lines.append(
                f"η² = {effect_size.get('etaSquared', 0.0):.4f}, "
                f"ω² = {effect_size.get('omegaSquared', 0.0):.4f}"
            )
        assumptions = r.get("assumptions", {})
        shapiro = "; ".join(
            f"{item['group']}: p={_p(item['p'])}"
            for item in assumptions.get("shapiroWilk", [])
            if item.get("p") is not None
        )
        levene = assumptions.get("levene", {})
        if shapiro:
            lines.append(f"Shapiro-Wilk normality checks: {shapiro}")
        if levene.get("p") is not None:
            lines.append(f"Levene variance check: W={levene['W']:.4f}, p={_p(levene['p'])}")
        post_hoc = r.get("postHoc")
        if post_hoc:
            comparisons = "; ".join(
                f"{item['group1']} vs {item['group2']}: p_adj={_p(item['pAdjusted'])}"
                f"{' (significant)' if item['reject'] else ''}"
                for item in post_hoc.get("comparisons", [])
            )
            if comparisons:
                lines.append(f"Tukey HSD post-hoc: {comparisons}")
        lines.append("Significant at α = .05." if r["isSignificant"] else "Not significant at α = .05.")
        return "\n".join(lines)
    if test == "two-way anova":
        lines = [f"{title}Two-way ANOVA ({r['factorA']} × {r['factorB']})"]
        lines.extend(f"{effect['effect']}: F({effect['df']:.0f}, {r['residualDf']:.0f}) = {effect['F']:.4f}, p = {_p(effect['p'])}" for effect in r["effects"])
        if r["interactionSignificant"]:
            lines.append("Interaction is significant; simple effects are needed.")
        return "\n".join(lines)
    if test == "simple linear regression":
        return (
            f"{title}Simple linear regression\n"
            f"{r['outcome']} = {r['intercept']:.4f} + {r['slope']:.4f} × {r['predictor']}\n"
            f"slope={r['slope']:.4f}, intercept={r['intercept']:.4f}, r={r['r']:.4f}, r²={r['rSquared']:.4f}, "
            f"p={_p(r['p'])}, n={r['n']}"
        )
    if test in {"pearson", "spearman"}:
        return (
            f"{title}{test.title()} correlation\n"
            f"r({r['n'] - 2}) = {r['r']:.4f}, p = {_p(r['p'])}, n = {r['n']}\n"
            f"{'Significant at α = .05.' if r['isSignificant'] else 'Not significant at α = .05.'}"
        )
    if test == "fisher-exact":
        return (
            f"{title}Fisher exact test (2x2). Also χ² uncorrected = {r['chi2Uncorrected']:.4f}, "
            f"p_chi = {_p(r['chi2pUncorrected'])}.\n"
            f"Fisher p = {_p(r['p'])}. "
            f"{'Significant at α = .05.' if r['isSignificant'] else 'Not significant at α = .05.'}"
        )
    if test == "chi-square":
        return f"{title}Chi-square: χ²({r['df']}) = {r['chi2']:.4f}, p = {_p(r['p'])}"
    return title + json.dumps(r, default=str)


def build_breakdown(engine: dict[str, Any], max_results: int | None = None) -> str:
    """Build one layman-English breakdown for the whole verified study."""
    results = engine.get("results") or ([engine["result"]] if engine.get("result") else [])
    if max_results is not None:
        results = results[:max_results]
    if not results:
        return "No verified study result was returned."

    lines = [
        _breakdown_synthesis_line(results, engine.get("topic")),
        _breakdown_significance_sentence(results),
        _breakdown_sample_caveat(results),
        _breakdown_limits(results),
    ]
    return "\n".join(lines)


def _breakdown_comparison(results: list[dict[str, Any]]) -> str:
    first = results[0]
    test = first.get("test")
    if test in {"student-t", "welch-t"}:
        names = list(dict.fromkeys(
            str(group["name"])
            for result in results
            for group in (result["group1"], result["group2"])
        ))
        return " and ".join(names) + " on " + ", ".join(_breakdown_label(result) for result in results)
    if test == "one-way anova":
        names = list(dict.fromkeys(
            str(group["name"])
            for result in results
            for group in result.get("groups", [])
        ))
        return ", ".join(names) + " on " + ", ".join(_breakdown_label(result) for result in results)
    if test == "paired-t":
        return "matched " + first["before"]["name"] + " and " + first["after"]["name"] + " values"
    if test in {"simple linear regression", "pearson", "spearman"}:
        return ", ".join(
            f"{result.get('predictor', result.get('xName', 'X'))} and "
            f"{result.get('outcome', result.get('yName', 'Y'))}"
            for result in results
        )
    if test == "two-way anova":
        return ", ".join(
            f"{result['factorA']}, {result['factorB']}, and their interaction"
            for result in results
        )
    return "the submitted count categories"


def _breakdown_outcome_line(result: dict[str, Any]) -> str:
    label = _breakdown_label(result)
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        first, second = result["group1"], result["group2"]
        higher, lower = sorted((first, second), key=lambda group: group["mean"], reverse=True)
        return (
            f"{label}: {higher['name']} was higher ({higher['mean']:.3f}) than "
            f"{lower['name']} ({lower['mean']:.3f}); t({_df(result['df'])}) = "
            f"{result['t']:.4f}, p = {_p(result['p'])}."
        )
    if test == "one-way anova":
        groups = result.get("groups", [])
        if groups:
            highest = max(groups, key=lambda group: group["mean"])
            lowest = min(groups, key=lambda group: group["mean"])
            direction = f"{highest['name']} was highest ({highest['mean']:.3f}) and {lowest['name']} was lowest ({lowest['mean']:.3f})"
        else:
            direction = "group direction was not reported"
        return (
            f"{label}: {direction}; F({result['dfb']}, {result['dfw']}) = "
            f"{result['F']:.4f}, p = {_p(result['p'])}."
        )
    if test == "paired-t":
        before, after = result["before"], result["after"]
        direction = "higher" if after["mean"] > before["mean"] else "lower" if after["mean"] < before["mean"] else "the same"
        return (
            f"{label}: {after['name']} was {direction} than {before['name']} "
            f"({after['mean']:.3f} versus {before['mean']:.3f}); t({_df(result['df'])}) = "
            f"{result['t']:.4f}, p = {_p(result['p'])}."
        )
    if test == "two-way anova":
        effects = "; ".join(
            f"{effect['effect']} F({effect['df']:.0f}, {result['residualDf']:.0f}) = "
            f"{effect['F']:.4f}, p = {_p(effect['p'])}"
            for effect in result.get("effects", [])
        )
        return f"{label}: effects were {effects or 'not reported'}."
    if test == "simple linear regression":
        direction = "higher" if result["slope"] >= 0 else "lower"
        return (
            f"{label}: higher {result['predictor']} went with {direction} {result['outcome']}; "
            f"r = {result['r']:.4f}, p = {_p(result['p'])}."
        )
    if test in {"pearson", "spearman"}:
        direction = "together" if result["r"] >= 0 else "in opposite directions"
        return (
            f"{label}: the columns moved {direction}; r = {result['r']:.4f}, "
            f"p = {_p(result['p'])}."
        )
    if test == "fisher-exact":
        return f"{label}: the categories showed an association; Fisher p = {_p(result['p'])}."
    if test == "chi-square":
        return (
            f"{label}: observed counts differed from chance expectations; χ²({result['df']}) = "
            f"{result['chi2']:.4f}, p = {_p(result['p'])}."
        )
    return f"{label}: the engine did not report a plain-English direction."


def _breakdown_significance_sentence(results: list[dict[str, Any]]) -> str:
    significant: list[str] = []
    not_significant: list[str] = []
    for result in results:
        label = _breakdown_label(result)
        if result.get("test") == "two-way anova":
            for effect in result.get("effects", []):
                target = f"the {effect['effect']} effect on {label}"
                (significant if effect.get("isSignificant") else not_significant).append(target)
        else:
            target = _breakdown_evidence_target(result)
            (significant if result.get("isSignificant") else not_significant).append(target)
    if significant and not_significant:
        return (
            f"At the 5% level, the data supported {_breakdown_join(significant)}, "
            f"but the evidence was not clear for {_breakdown_join(not_significant)}."
        )
    if significant:
        return f"At the 5% level, the data supported {_breakdown_join(significant)}."
    return (
        "The observed pattern may be meaningful, but the tests did not provide clear "
        "statistical evidence at the 5% level."
    )


def _breakdown_evidence_target(result: dict[str, Any]) -> str:
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
    return f"the difference in {_breakdown_label(result)}"


def _breakdown_synthesis_line(results: list[dict[str, Any]], topic: Any) -> str:
    if results and all(result.get("test") == "paired-t" for result in results):
        story = "; ".join(
            _paired_breakdown_fragment(result) for result in results
        ) + "."
        context = (
            f"Taken together for {str(topic).strip()}, "
            if str(topic or "").strip()
            else "Taken together, "
        )
        return context + story
    group_directions = [_breakdown_group_direction(result) for result in results]
    group_directions = [item for item in group_directions if item]
    other_results = [
        result
        for result in results
        if _breakdown_group_direction(result) is None
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
            high_text = _breakdown_direction_story(
                standouts[0],
                high_groups[standouts[0]],
                "higher",
            )
        else:
            high_text = "; ".join(
                _breakdown_direction_story(name, high_groups[name], "higher")
                for name in standouts
            )
        story = (
            high_text + "; "
            + "; ".join(
                _breakdown_direction_story(name, labels, "lower")
                for name, labels in low_groups.items()
            )
        )
        if other_results:
            story += "; " + "; ".join(
                _breakdown_synthesis_fragment(result) for result in other_results
            )
    else:
        story = "; ".join(_breakdown_synthesis_fragment(result) for result in results)
    context = f"Taken together for {str(topic).strip()}, " if str(topic or "").strip() else "Taken together, "
    return context + story + "."


def _breakdown_direction_story(name: str, labels: list[str], direction: str) -> str:
    return f"{name} " + _breakdown_join([
        _breakdown_direction_phrase(label, direction) for label in labels
    ])


def _breakdown_direction_phrase(label: str, direction: str) -> str:
    clean = str(label).strip()
    lower = clean.lower()
    if re.search(r"\bph\b", clean, flags=re.I):
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


def _paired_breakdown_fragment(result: dict[str, Any]) -> str:
    before, after = result["before"], result["after"]
    if after["mean"] > before["mean"]:
        direction = "higher than"
    elif after["mean"] < before["mean"]:
        direction = "lower than"
    else:
        direction = "the same as"
    suffix = f" on {_breakdown_label(result)}" if result.get("parameter") else ""
    return f"{after['name']} was {direction} {before['name']}{suffix}"


def _breakdown_group_direction(result: dict[str, Any]) -> tuple[str, str, str] | None:
    label = _breakdown_label(result)
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        higher, lower = sorted(
            (result["group1"], result["group2"]),
            key=lambda group: group["mean"],
            reverse=True,
        )
        return label, str(higher["name"]), str(lower["name"])
    if test == "one-way anova" and result.get("groups"):
        higher = max(result["groups"], key=lambda group: group["mean"])
        lower = min(result["groups"], key=lambda group: group["mean"])
        return label, str(higher["name"]), str(lower["name"])
    if test == "paired-t":
        return label, str(result["after"]["name"]), str(result["before"]["name"])
    return None


def _breakdown_synthesis_fragment(result: dict[str, Any]) -> str:
    label = _breakdown_label(result)
    test = result.get("test")
    if test == "simple linear regression":
        return (
            f"{label} rose with {result['predictor']}"
            if result["slope"] >= 0
            else f"{label} fell as {result['predictor']} rose"
        )
    if test in {"pearson", "spearman"}:
        return f"{label} rose with its paired column" if result["r"] >= 0 else f"{label} moved against its paired column"
    if test == "two-way anova":
        effects = ", ".join(str(effect["effect"]) for effect in result.get("effects", []))
        return f"{label} followed the tested {effects or 'factor'} pattern"
    return f"{label} changed across the submitted data"


def _breakdown_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _breakdown_sample_once(results: list[dict[str, Any]]) -> str:
    groups: list[tuple[str, Any]] = []
    other: list[str] = []
    for result in results:
        test = result.get("test")
        if test in {"student-t", "welch-t"}:
            groups.extend(
                (str(group["name"]), group["n"])
                for group in (result["group1"], result["group2"])
            )
        elif test == "one-way anova":
            groups.extend(
                (str(group["name"]), group["n"]) for group in result.get("groups", [])
            )
        elif test == "paired-t":
            other.append(f"{result['nPairs']} matched pairs")
        elif result.get("n") is not None:
            other.append(f"n={result['n']}")
    groups = list(dict.fromkeys(groups))
    sizes = list(dict.fromkeys(size for _, size in groups))
    if groups and len(sizes) == 1:
        return f"n = {sizes[0]} per group ({', '.join(name for name, _ in groups)})"
    if groups:
        return "; ".join(f"n = {size} for {name}" for name, size in groups)
    return "; ".join(dict.fromkeys(other)) or "sample size was not reported"


def _breakdown_limits(results: list[dict[str, Any]]) -> str:
    clauses = []
    if _breakdown_has_sample_size(results):
        clauses.append("Small n limits how widely this pattern can be generalized")
    if any(result.get("test") in {"one-way anova", "two-way anova"} for result in results):
        clauses.append("ANOVA does not establish that every pair of groups differs")
    clauses.append("No causation is established")
    clauses.append(_breakdown_outcome_limit(results))
    return "Limits: " + ". ".join(clauses) + "."


def _breakdown_outcome_limit(results: list[dict[str, Any]]) -> str:
    labels = [_breakdown_label(result) for result in results]
    label_text = _breakdown_join(labels)
    lower_labels = " ".join(labels).lower()
    if re.search(r"\bdwell\b|\btime\b|\bduration\b|\bseconds?\b|\bminutes?\b", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe a longer or shorter duration only; "
            "they do not by themselves establish product quality, safety, approval, or market suitability"
        )
    if re.search(r"\bph\b", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe acidity only; "
            "they do not by themselves establish product quality, safety, approval, or market suitability"
        )
    if re.search(r"\bmoisture\b|\bwater(?:\s+content)?\b|water_content", lower_labels):
        return (
            f"Higher or lower values on {label_text} describe moisture only; "
            "they do not by themselves establish product quality, safety, approval, or market suitability"
        )
    return (
        f"Higher or lower values on {label_text} describe the measured response only; "
        "they do not by themselves establish product quality, safety, approval, or market suitability"
    )


def _breakdown_sample_caveat(results: list[dict[str, Any]]) -> str:
    count_total = _breakdown_count_total(results)
    if count_total is not None:
        return f"Count total: {count_total}; counts are not replicate measurements."
    sample = _breakdown_sample_once(results)
    if sample == "sample size was not reported":
        return "n caveat: sample size was not reported."
    return f"n caveat: {sample}; this is a small sample, not a huge trial."


def _breakdown_has_sample_size(results: list[dict[str, Any]]) -> bool:
    return _breakdown_count_total(results) is not None or any(
        result.get("n") is not None
        or result.get("nPairs") is not None
        or any(group.get("n") is not None for group in result.get("groups", []))
        or any(
            isinstance(result.get(key), dict) and result[key].get("n") is not None
            for key in ("before", "after", "group1", "group2")
        )
        for result in results
    )


def _breakdown_count_total(results: list[dict[str, Any]]) -> int | None:
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


def _breakdown_has_moisture_outcome(results: list[dict[str, Any]]) -> bool:
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


def _breakdown_practical_line(results: list[dict[str, Any]]) -> str:
    labels = " ".join(_breakdown_label(result).lower() for result in results)
    if "moisture" in labels:
        return "Plain reading: for a moisture outcome, lower values mean a drier product; that is not automatically safer or better."
    return "Plain reading: higher or lower values mean more or less of the named outcome; that is not automatically better."


def _breakdown_label(result: dict[str, Any]) -> str:
    if result.get("parameter") or result.get("outcome"):
        return str(result.get("parameter") or result.get("outcome"))
    if result.get("test") == "paired-t":
        return f"{result.get('before', {}).get('name', 'before')} vs {result.get('after', {}).get('name', 'after')}"
    if result.get("test") in {"fisher-exact", "chi-square"}:
        return "count categories"
    return "the submitted outcome"


def _breakdown_test_name(result: dict[str, Any]) -> str:
    return {
        "student-t": "independent t-test",
        "welch-t": "Welch t-test",
        "paired-t": "paired t-test",
        "one-way anova": "one-way ANOVA",
        "two-way anova": "two-way ANOVA",
        "simple linear regression": "simple linear regression",
        "pearson": "Pearson correlation",
        "spearman": "Spearman correlation",
        "fisher-exact": "Fisher exact test",
        "chi-square": "chi-square test",
    }.get(str(result.get("test")), str(result.get("test") or "selected test"))


def _breakdown_statistic(result: dict[str, Any]) -> str:
    test = result.get("test")
    label = _breakdown_label(result)
    if test in {"student-t", "welch-t", "paired-t"}:
        return f"{label}: t({_df(result['df'])}) = {result['t']:.4f}, p = {_p(result['p'])}"
    if test == "one-way anova":
        return f"{label}: F({result['dfb']}, {result['dfw']}) = {result['F']:.4f}, p = {_p(result['p'])}"
    if test == "two-way anova":
        effects = ", ".join(
            f"{effect['effect']} F={effect['F']:.4f}, p={_p(effect['p'])}"
            for effect in result.get("effects", [])
        )
        return f"{label}: {effects or 'effects not reported'}"
    if test == "simple linear regression":
        return f"{label}: r={result['r']:.4f}, r²={result['rSquared']:.4f}, p={_p(result['p'])}"
    if test in {"pearson", "spearman"}:
        return f"{label}: r={result['r']:.4f}, p={_p(result['p'])}"
    if test == "fisher-exact":
        return f"{label}: Fisher p={_p(result['p'])}"
    if test == "chi-square":
        return f"{label}: χ²({result['df']})={result['chi2']:.4f}, p={_p(result['p'])}"
    return f"{label}: statistic not reported"


def _breakdown_decision(result: dict[str, Any]) -> str:
    label = _breakdown_label(result)
    if result.get("test") == "two-way anova":
        significant = [
            str(effect.get("effect"))
            for effect in result.get("effects", [])
            if effect.get("isSignificant")
        ]
        decision = f"significant effect(s): {', '.join(significant)}" if significant else "no significant effects"
    else:
        decision = "significant at α = .05" if result.get("isSignificant") else "not significant at α = .05"
    return f"{label} {decision}"


def _breakdown_sample(result: dict[str, Any]) -> str:
    test = result.get("test")
    label = _breakdown_label(result)
    if test in {"student-t", "welch-t"}:
        return f"{label} n={result['group1']['n']} and n={result['group2']['n']}"
    if test == "one-way anova":
        return f"{label} " + ", ".join(f"{group['name']} n={group['n']}" for group in result.get("groups", []))
    if test == "paired-t":
        return f"{label} {result['nPairs']} matched pairs"
    return f"{label} n={result.get('n', 'not reported')}"


def _breakdown_result(result: dict[str, Any]) -> str:
    test = result.get("test")
    label = result.get("parameter") or result.get("outcome") or "the measured outcome"
    if test in {"student-t", "welch-t"}:
        a, b = result["group1"], result["group2"]
        higher, lower = (a, b) if a["mean"] >= b["mean"] else (b, a)
        trust = "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance."
        return "\n".join([
            f"What was compared: {a['name']} and {b['name']} for {label}.",
            f"{higher['name']} came out higher ({higher['mean']:.4f}) than {lower['name']} ({lower['mean']:.4f}).",
            f"Test result: t({_df(result['df'])}) = {result['t']:.4f}, p = {_p(result['p'])}.",
            trust,
            f"Difference size (Cohen's d) = {result['cohensD']:.4f}.",
            f"Samples: n={a['n']} and n={b['n']}; this is a signal, not a huge trial.",
            "This does not prove that one method is better or safer for every situation.",
        ])
    if test == "paired-t":
        higher = result["after"]["name"] if result["after"]["mean"] >= result["before"]["mean"] else result["before"]["name"]
        trust = "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance."
        return "\n".join([
            f"What was compared: matched {result['before']['name']} and {result['after']['name']} values for {label}.",
            f"{higher} had the higher average; the after-minus-before difference was {result['meanDifference']:.4f}.",
            f"Test result: t({_df(result['df'])}) = {result['t']:.4f}, p = {_p(result['p'])}.",
            trust,
            f"Difference size (Cohen's dz) = {result['cohensDz']:.4f}.",
            f"Samples: {result['nPairs']} matched pairs; this is a signal, not a huge trial.",
            "This does not prove that the change will happen for every person or setting.",
        ])
    if test == "one-way anova":
        higher = max(result["groups"], key=lambda group: group["mean"])
        sample_sizes = ", ".join(f"{group['name']} n={group['n']}" for group in result["groups"])
        trust = "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance."
        return "\n".join([
            f"What was compared: {', '.join(str(group['name']) for group in result['groups'])} for {label}.",
            f"{higher['name']} had the highest average ({higher['mean']:.4f}).",
            f"Test result: F({result['dfb']}, {result['dfw']}) = {result['F']:.4f}, p = {_p(result['p'])}.",
            trust,
            f"Samples: {sample_sizes}; small samples are a signal, not a huge trial.",
            "This result does not prove that every pair of groups is different.",
            "It also does not prove that the highest group is better for every situation.",
        ])
    if test == "two-way anova":
        effects = result["effects"]
        lines = [
            f"What was compared: {result['factorA']}, {result['factorB']}, and their interaction for {result['outcome']}.",
            *[f"{effect['effect']}: F({effect['df']:.0f}, {result['residualDf']:.0f}) = {effect['F']:.4f}, p = {_p(effect['p'])}." for effect in effects],
            "At least one effect is unlikely to be only luck." if result["isSignificant"] else "The tested effects could still be chance.",
            f"Samples: n={result['n']}; this is a signal, not a huge trial.",
            "A significant interaction means simple effects are needed." if result["interactionSignificant"] else "The interaction does not require a simple-effects warning.",
            "This does not prove that one factor combination is best for every situation.",
        ]
        return "\n".join(lines)
    if test == "simple linear regression":
        trust = "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance."
        return "\n".join([
            f"What was compared: {result['predictor']} as one predictor of {result['outcome']}.",
            f"Higher {result['predictor']} values went with {'higher' if result['slope'] >= 0 else 'lower'} {result['outcome']} values.",
            f"Fitted slope = {result['slope']:.4f}; intercept = {result['intercept']:.4f}.",
            f"Test result: r = {result['r']:.4f}, r² = {result['rSquared']:.4f}, p = {_p(result['p'])}.",
            trust,
            f"Explained share (r²) = {result['rSquared']:.4f}.",
            f"Samples: n={result['n']}; this is a signal, not a huge trial.",
            "This does not prove that the predictor causes the outcome or predicts every case.",
        ])
    if test in {"pearson", "spearman"}:
        trust = "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance."
        return "\n".join([
            f"What was compared: {result.get('xName', 'X')} and {result.get('yName', 'Y')}.",
            f"The columns moved {'together' if result['r'] >= 0 else 'in opposite directions'} (r = {result['r']:.4f}).",
            f"Test result: p = {_p(result['p'])}.",
            trust,
            f"Relationship size: r = {result['r']:.4f}.",
            f"Samples: n={result['n']}; this is a signal, not a huge trial.",
            "This does not prove that either column causes the other.",
        ])
    if test == "fisher-exact":
        return "\n".join([
            "What was compared: the two-by-two count table.",
            "The table compares how often each category occurred.",
            f"Test result: Fisher p = {_p(result['p'])}; uncorrected chi-square p = {_p(result['chi2pUncorrected'])}.",
            "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance.",
            f"Association size (odds ratio) = {result['oddsRatio']:.4f}.",
            "The table contains counts, not replicate measurements.",
            "This does not prove that one category causes the other.",
        ])
    if test == "chi-square":
        return "\n".join([
            "What was compared: the observed count table against expected counts.",
            "The table compares observed counts with what would be expected by chance.",
            f"Test result: χ²({result['df']}) = {result['chi2']:.4f}, p = {_p(result['p'])}.",
            "This is unlikely to be only luck." if result["isSignificant"] else "This could still be chance.",
            "The table contains counts, not replicate measurements.",
            "This does not prove that one category causes the other.",
        ])
    return "No plain-English breakdown is available for this engine result."


def _p(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.4f}"


def _df(df: float) -> str:
    return str(int(df)) if abs(df - round(df)) < 1e-6 else f"{df:.2f}"


def _requested_mode(text: str, request: dict) -> str | None:
    if request.get("mode") in {"correlation", "regression"}:
        return request["mode"]
    lower = text.lower()
    if "correlation" in lower or "pearson" in lower or "spearman" in lower:
        return "correlation"
    return "regression" if "regression" in lower else None


def _requested_outcome(request: dict) -> str | None:
    requested = request.get("outcome") or request.get("outcome_name")
    if requested is None:
        return None
    return str(requested).strip() or None


def _is_unsupported(text: str) -> bool:
    lower = text.lower()
    if any(term in lower for term in ("forecast", "forecasting", "mixed model", "survival analysis", "generalized linear", "glm", "logistic regression", "multiple regression")):
        return True
    return bool(re.search(r"\b\d+\s+predictors?\b", lower))


if __name__ == "__main__":
    import sys

    text = sys.stdin.read() if not sys.argv[1:] else " ".join(sys.argv[1:])
    print(json.dumps(handle_analyze({"text": text}), indent=2, default=str))