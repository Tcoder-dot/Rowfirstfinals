"""Deterministic stats engine. LLM must not compute these numbers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _mean_sd(x: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(x))
    return mean, float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def _ci_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> list[float]:
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    se = sp * math.sqrt(1 / n1 + 1 / n2)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n1 + n2 - 2))
    return [diff - tcrit * se, diff + tcrit * se]


def _ci_diff_welch(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> list[float]:
    n1, n2 = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    return [diff - tcrit * se, diff + tcrit * se]


def describe_group(name: str, values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    mean, sd = _mean_sd(x)
    sw = None
    if 3 <= len(x) <= 5000 and np.std(x) > 0:
        W, p = stats.shapiro(x)
        sw = {"W": float(W), "p": float(p), "isNormal": bool(p >= 0.05)}
    return {"name": name, "n": int(len(x)), "mean": mean, "sd": sd, "median": float(np.median(x)), "shapiroWilk": sw}


def independent_ttest(g1: dict, g2: dict, equal_var: bool = True) -> dict[str, Any]:
    a = np.asarray(g1["values"], dtype=float)
    b = np.asarray(g2["values"], dtype=float)
    res = stats.ttest_ind(a, b, equal_var=equal_var)
    d = _cohens_d(a, b)
    df = float(res.df) if hasattr(res, "df") and res.df is not None else (len(a) + len(b) - 2)
    return {
        "test": "welch-t" if not equal_var else "student-t",
        "group1": describe_group(g1["name"], g1["values"]),
        "group2": describe_group(g2["name"], g2["values"]),
        "t": float(res.statistic),
        "df": float(df),
        "p": float(res.pvalue),
        "difference": float(np.mean(a) - np.mean(b)),
        "confidenceInterval95": _ci_diff(a, b) if equal_var else _ci_diff_welch(a, b),
        "cohensD": d,
        "isSignificant": bool(res.pvalue < 0.05),
    }


def _anova_effect_sizes(arrs: list[np.ndarray]) -> dict[str, float]:
    """Return eta-squared and omega-squared for a one-way ANOVA."""
    all_values = np.concatenate(arrs)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in arrs))
    ss_within = float(sum(np.sum((values - np.mean(values)) ** 2) for values in arrs))
    ss_total = ss_between + ss_within
    df_within = sum(len(values) for values in arrs) - len(arrs)
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    eta_squared = ss_between / ss_total if ss_total else 0.0
    omega_denominator = ss_total + ms_within
    omega_squared = (
        (ss_between - (len(arrs) - 1) * ms_within) / omega_denominator
        if omega_denominator
        else 0.0
    )
    return {
        "etaSquared": float(eta_squared),
        "omegaSquared": float(omega_squared),
    }


def _anova_assumptions(groups: list[dict], arrs: list[np.ndarray]) -> dict[str, Any]:
    shapiro = []
    for group, values in zip(groups, arrs):
        result = describe_group(str(group["name"]), values.tolist()).get("shapiroWilk")
        shapiro.append({
            "group": str(group["name"]),
            **(result or {"available": False, "reason": "Shapiro-Wilk needs 3–5000 non-constant observations."}),
        })

    levene_result: dict[str, Any]
    if len(arrs) >= 2 and all(len(values) >= 2 for values in arrs):
        statistic, p_value = stats.levene(*arrs, center="median")
        levene_result = {
            "W": float(statistic),
            "p": float(p_value),
            "equalVariance": bool(p_value >= 0.05),
        }
    else:
        levene_result = {
            "available": False,
            "reason": "Levene's test needs at least two observations per group.",
        }
    return {"shapiroWilk": shapiro, "levene": levene_result}


def _tukey_post_hoc(groups: list[dict], arrs: list[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate(arrs)
    labels = np.concatenate([
        np.repeat(str(group["name"]), len(values_for_group))
        for group, values_for_group in zip(groups, arrs)
    ])
    tukey = pairwise_tukeyhsd(values, labels, alpha=0.05)
    comparisons = []
    for row in tukey.summary().data[1:]:
        group1, group2, mean_difference, p_adjusted, lower, upper, reject = row
        comparisons.append({
            "group1": str(group1),
            "group2": str(group2),
            "meanDifference": float(mean_difference),
            "pAdjusted": float(p_adjusted),
            "lower": float(lower),
            "upper": float(upper),
            "reject": bool(reject),
        })
    return {
        "test": "Tukey HSD",
        "alpha": 0.05,
        "comparisons": comparisons,
    }


def one_way_anova(groups: list[dict], outcome: str | None = None) -> dict[str, Any]:
    arrs = [np.asarray(g["values"], dtype=float) for g in groups]
    res = stats.f_oneway(*arrs)
    k = len(groups)
    n = sum(len(a) for a in arrs)
    result = {
        "test": "one-way anova",
        "groups": [describe_group(g["name"], g["values"]) for g in groups],
        "F": float(res.statistic),
        "dfb": k - 1,
        "dfw": n - k,
        "p": float(res.pvalue),
        "isSignificant": bool(res.pvalue < 0.05),
        "assumptions": _anova_assumptions(groups, arrs),
        "effectSize": _anova_effect_sizes(arrs),
    }
    result["shapiroWilk"] = result["assumptions"]["shapiroWilk"]
    result["levene"] = result["assumptions"]["levene"]
    result["etaSquared"] = result["effectSize"]["etaSquared"]
    result["omegaSquared"] = result["effectSize"]["omegaSquared"]
    if outcome:
        result["outcome"] = str(outcome)
    if result["isSignificant"]:
        result["postHoc"] = _tukey_post_hoc(groups, arrs)
    return result


def chi_or_fisher(matrix: list[list[int]]) -> dict[str, Any]:
    m = np.asarray(matrix, dtype=float)
    if m.shape == (2, 2):
        oddsr, p = stats.fisher_exact(m)
        chi = stats.chi2_contingency(m, correction=False)
        return {
            "test": "fisher-exact",
            "matrix": matrix,
            "oddsRatio": float(oddsr),
            "p": float(p),
            "chi2Uncorrected": float(chi.statistic),
            "chi2pUncorrected": float(chi.pvalue),
            "isSignificant": bool(p < 0.05),
        }
    chi = stats.chi2_contingency(m, correction=False)
    return {
        "test": "chi-square",
        "matrix": matrix,
        "chi2": float(chi.statistic),
        "df": int(chi.dof),
        "p": float(chi.pvalue),
        "expected": chi.expected_freq.tolist(),
        "isSignificant": bool(chi.pvalue < 0.05),
    }


def correlation(x: list[float], y: list[float], method: str = "pearson") -> dict[str, Any]:
    """Compute a correlation from two numeric columns without any LLM arithmetic."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Correlation needs at least three paired observations.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation columns must contain finite numbers only.")
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be pearson or spearman.")
    result = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    return {
        "test": method,
        "n": int(len(a)),
        "r": float(result.statistic),
        "p": float(result.pvalue),
        "isSignificant": bool(result.pvalue < 0.05),
    }


def paired_ttest(
    before: list[float], after: list[float], pair_ids: list[str] | None = None
) -> dict[str, Any]:
    """Paired t-test and Cohen's dz from already matched observations."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired t-test needs at least two matched pairs.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired columns must contain finite numbers only.")
    differences = b - a
    result = stats.ttest_rel(a, b)
    sd_diff = float(np.std(differences, ddof=1))
    dz = float(np.mean(differences) / sd_diff) if sd_diff else 0.0
    return {
        "test": "paired-t",
        "nPairs": int(len(a)),
        "before": {"name": "before", "n": int(len(a)), "mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))},
        "after": {"name": "after", "n": int(len(b)), "mean": float(np.mean(b)), "sd": float(np.std(b, ddof=1))},
        "meanDifference": float(np.mean(differences)),
        "t": float(result.statistic),
        "df": float(len(a) - 1),
        "p": float(result.pvalue),
        "cohensDz": dz,
        "isSignificant": bool(result.pvalue < 0.05),
        "pairIds": pair_ids or [str(i + 1) for i in range(len(a))],
        "pairs": list(zip(a.tolist(), b.tolist())),
    }


def two_way_anova(
    rows: list[dict[str, Any]], factor_a: str, factor_b: str, outcome: str
) -> dict[str, Any]:
    """Type-II two-way ANOVA using statsmodels for the design matrix and tests."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame = frame[[factor_a, factor_b, outcome]].dropna()
    if len(frame) < 4:
        raise ValueError("Two-way ANOVA needs at least four complete observations.")
    frame[outcome] = pd.to_numeric(frame[outcome], errors="raise")
    frame["_a"] = frame[factor_a].astype(str)
    frame["_b"] = frame[factor_b].astype(str)
    frame["_y"] = frame[outcome].astype(float)
    if frame["_a"].nunique() < 2 or frame["_b"].nunique() < 2:
        raise ValueError("Both two-way ANOVA factors need at least two levels.")
    model = ols("_y ~ C(_a) * C(_b)", data=frame).fit()
    table = anova_lm(model, typ=2)
    residual_df = float(table.loc["Residual", "df"])
    effects = []
    labels = [
        (f"{factor_a}", "C(_a)"),
        (f"{factor_b}", "C(_b)"),
        (f"{factor_a}:{factor_b}", "C(_a):C(_b)"),
    ]
    for label, row_name in labels:
        row = table.loc[row_name]
        effects.append(
            {
                "effect": label,
                "df": float(row["df"]),
                "F": float(row["F"]),
                "p": float(row["PR(>F)"]),
                "isSignificant": bool(row["PR(>F)"] < 0.05),
            }
        )
    return {
        "test": "two-way anova",
        "factorA": factor_a,
        "factorB": factor_b,
        "outcome": outcome,
        "n": int(len(frame)),
        "residualDf": residual_df,
        "effects": effects,
        "interactionSignificant": bool(effects[2]["p"] < 0.05),
        "isSignificant": bool(any(effect["p"] < 0.05 for effect in effects)),
    }


def linear_regression(x: list[float], y: list[float], x_name: str = "X", y_name: str = "Y") -> dict[str, Any]:
    """One-predictor linear regression from scipy.stats.linregress."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Simple linear regression needs at least three paired observations.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Regression columns must contain finite numbers only.")
    result = stats.linregress(a, b)
    return {
        "test": "simple linear regression",
        "predictor": x_name,
        "outcome": y_name,
        "n": int(len(a)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "rSquared": float(result.rvalue ** 2),
        "p": float(result.pvalue),
        "stderr": float(result.stderr),
        "isSignificant": bool(result.pvalue < 0.05),
        "pairs": list(zip(a.tolist(), b.tolist())),
    }


def analyze_groups(
    groups: list[dict],
    design: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    groups = [g for g in groups if g.get("values")]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with numeric values.")
    if design == "anova" or (design is None and len(groups) >= 3):
        return one_way_anova(groups, outcome)
    if len(groups) == 2:
        a, b = groups[0]["values"], groups[1]["values"]
        equal = True
        if len(a) >= 3 and len(b) >= 3:
            w, p = stats.levene(a, b, center="median")
            equal = p >= 0.05
        return independent_ttest(groups[0], groups[1], equal_var=equal)
    return one_way_anova(groups, outcome)
