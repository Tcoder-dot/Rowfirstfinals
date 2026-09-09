"""Turn student text into groups. Ignore Sample_ID. Use Treatment when present."""
from __future__ import annotations

import csv
import io
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
ID_HEADERS = {
    "sample_id", "sampleid", "sample", "id", "replicate", "rep", "jar", "code",
    "user_id", "userid", "subject", "subject_id", "patient", "patient_id",
}
GROUP_HEADERS = {"treatment", "treatments", "group", "groups", "method", "methods", "condition", "arm", "variant"}
TIME_HEADERS = {"time", "visit", "phase", "occasion", "period", "stage"}
FACTOR_HINTS = GROUP_HEADERS | TIME_HEADERS | {"day", "plant", "block", "batch", "site", "location", "sex", "type", "category"}


def parse_numbers(cell: str) -> list[float]:
    if cell is None:
        return []
    return [float(x) for x in NUMBER_RE.findall(str(cell).replace(",", " "))]


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(h).lower())


def collapse_sample_id(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"[\s_\-]+[A-Za-z](?:[\s_\-]*r)?\s*\d+$", "", s, flags=re.I)
    s = re.sub(r"[\s_\-]*r\s*\d+$", "", s, flags=re.I)
    s = re.sub(r"[\s_\-]*rep(?:licate)?\s*\d+$", "", s, flags=re.I)
    s = re.sub(r"_+", " ", s).strip()
    if not s:
        return str(label).strip()
    return s


def ingest_text(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty input.")

    matrix = _maybe_count_matrix(raw)
    if matrix is not None:
        return {"format": "contingency", "matrix": matrix, "groups": []}

    rows, headers = _to_table(raw)
    if headers and rows:
        return ingest_table(rows, headers)

    groups = _labelled_lists(raw)
    if len(groups) >= 2:
        return {"format": "labelled", "groups": groups}

    raise ValueError("Could not find two groups of numbers. Use 'Brining: 1, 2, 3' or a table with a Treatment column.")


def ingest_file(path: str | os.PathLike[str]) -> dict:
    """Read supported table files locally. Archives use the first CSV/XLSX member only."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source.name}")
    suffix = source.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in {".csv", ".xlsx"}
            ]
            if not members:
                raise ValueError("ZIP must contain a CSV or XLSX file.")
            selected = members[0]
            with tempfile.TemporaryDirectory(prefix="rowfirst-") as tmp:
                extracted = Path(tmp) / Path(selected.filename).name
                extracted.write_bytes(archive.read(selected))
                return ingest_file(extracted)
    if suffix in {".txt", ".tsv"}:
        return ingest_text(source.read_text(encoding="utf-8-sig"))
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
        headers_row, headers = _find_header(rows)
        if headers is None:
            raise ValueError("Could not find a header row in the first 15 rows.")
        return ingest_table(headers_row, headers)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, header=None, dtype=object)
    else:
        raise ValueError("Supported local files are .csv, .txt, .tsv, .xlsx, .xls, or .zip containing CSV/XLSX.")
    return ingest_frame(frame, source.name)


def ingest_frame(frame: pd.DataFrame, source_name: str = "table") -> dict:
    """Find a usable header within the first 15 rows, then use the shared table rules."""
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"{source_name} is empty.")
    values = [[("" if pd.isna(v) else str(v).strip()) for v in row] for row in frame.values.tolist()]
    rows, headers = _find_header(values)
    if headers is None:
        raise ValueError("Could not find a header row in the first 15 rows.")
    return ingest_table(rows, headers)


def _maybe_count_matrix(text: str) -> list[list[int]] | None:
    # Delimited data belongs to the table parser, even when some columns are numeric.
    if any(delimiter in text for delimiter in (",", ";", "\t", "|")):
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    grid = []
    for ln in lines:
        nums = [int(float(x)) for x in NUMBER_RE.findall(ln)]
        if len(nums) >= 2:
            grid.append(nums)
    if len(grid) >= 2 and len({len(r) for r in grid}) == 1:
        if all(len(r) >= 2 for r in grid) and all(all(n >= 0 for n in r) for r in grid):
            # only treat as counts if there are no group labels like "Brining:"
            if not re.search(r":\s*\d", text) and len(grid) <= 8:
                # reject if it looks like two sample lists of floats
                if all(all(float(n).is_integer() for n in r) for r in grid):
                    return grid
    return None


def _to_table(text: str) -> tuple[list[list[str]], list[str] | None]:
    sample = text.strip()
    dialect_text = sample
    try:
        dialect = csv.Sniffer().sniff(dialect_text.splitlines()[0] + "\n" + "\n".join(dialect_text.splitlines()[1:3]), delimiters=",;\t|")
        reader = csv.reader(io.StringIO(sample), dialect)
        rows = [r for r in reader if any(c.strip() for c in r)]
    except Exception:
        rows = []
        for ln in sample.splitlines():
            if not ln.strip() or set(ln.strip()) <= set("-+|"):
                continue
            if "|" in ln:
                rows.append([c.strip() for c in ln.split("|") if c.strip() != ""])
            elif "\t" in ln:
                rows.append([c.strip() for c in ln.split("\t")])
            elif "," in ln:
                rows.append([c.strip() for c in ln.split(",")])
    if not rows:
        return [], None
    return _find_header(rows)


def _find_header(rows: list[list[str]]) -> tuple[list[list[str]], list[str] | None]:
    """Find a header row among the first 15 non-empty rows, not just row zero."""
    if not rows:
        return [], None
    limit = min(15, len(rows))
    best: tuple[int, int] | None = None
    for index in range(limit):
        header = rows[index]
        if len(header) < 2:
            continue
        norm = [_norm(h) for h in header]
        has_factor = any(h in GROUP_HEADERS for h in norm)
        has_id = any(h in ID_HEADERS for h in norm)
        numeric_following = 0
        for candidate in rows[index + 1:index + 6]:
            numeric_following += sum(bool(parse_numbers(cell)) for cell in candidate)
        text_cells = sum(parse_numbers(cell) == [] for cell in header)
        score = (100 if has_factor else 30 if has_id else 0) + text_cells * 3 + numeric_following
        if has_factor or (numeric_following and text_cells >= max(2, len(header) // 2)):
            if best is None or score > best[1]:
                best = (index, score)
    if best is None:
        header = rows[0]
        header_like = sum(1 for c in header if parse_numbers(c) == []) >= max(1, len(header) // 2)
        return (rows[1:], header) if header_like else (rows, None)
    index = best[0]
    return rows[index + 1:], rows[index]


def ingest_table(rows: list[list[str]], headers: list[str] | None) -> dict:
    headers = headers or [f"col{i}" for i in range(len(rows[0]))]
    norm = [_norm(h) for h in headers]

    treat_idx = next((i for i, h in enumerate(norm) if h in GROUP_HEADERS), None)
    id_idx = next((i for i, h in enumerate(norm) if h in ID_HEADERS), None)
    id_indices = [i for i, h in enumerate(norm) if h in ID_HEADERS]
    num_idx = [
        i for i, h in enumerate(headers)
        if i not in id_indices and _column_numeric(rows, i) and not _is_constant_column(rows, i)
    ]
    composition_indices = _composition_only_indices(rows, headers, num_idx)
    if composition_indices and len(composition_indices) < len(num_idx):
        num_idx = [i for i in num_idx if i not in composition_indices]

    paired = _detect_paired(rows, headers, norm, num_idx, id_indices)
    if paired is not None:
        return paired

    factor_indices = [
        i for i, h in enumerate(norm)
        if i not in id_indices and _has_two_levels(rows, i)
        and (h in FACTOR_HINTS or not _column_numeric(rows, i))
    ]
    outcome_idx_candidates = [i for i in num_idx if i not in factor_indices]
    if len(factor_indices) == 2 and len(outcome_idx_candidates) == 1:
        a_idx, b_idx, outcome_idx = factor_indices[0], factor_indices[1], outcome_idx_candidates[0]
        two_way_rows = [
            {
                headers[a_idx]: r[a_idx].strip() if a_idx < len(r) else "",
                headers[b_idx]: r[b_idx].strip() if b_idx < len(r) else "",
                headers[outcome_idx]: parse_numbers(r[outcome_idx])[0] if outcome_idx < len(r) and parse_numbers(r[outcome_idx]) else None,
            }
            for r in rows
        ]
        two_way_rows = [r for r in two_way_rows if r[headers[a_idx]] and r[headers[b_idx]] and r[headers[outcome_idx]] is not None]
        return {
            "format": "two-way",
            "factorA": headers[a_idx],
            "factorB": headers[b_idx],
            "outcome": headers[outcome_idx],
            "rows": two_way_rows,
        }
    if len(factor_indices) > 2 and len(outcome_idx_candidates) == 1:
        return {
            "format": "needs-clarification",
            "question": "I found more than two possible factor columns. Which two factors should be used for the analysis?",
        }

    outcome_indices = outcome_idx_candidates if outcome_idx_candidates else num_idx
    if treat_idx is not None and outcome_indices:
        outcomes = []
        for j in outcome_indices:
            buckets: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                if j >= len(r) or treat_idx >= len(r):
                    continue
                name = collapse_sample_id(r[treat_idx])
                vals = parse_numbers(r[j])
                if name and vals:
                    buckets[name].extend(vals)
            groups = [{"name": k, "values": v} for k, v in buckets.items() if len(v) >= 1]
            if len(groups) >= 2:
                outcomes.append({"parameter": headers[j], "groups": groups})
        if outcomes:
            return {"format": "long-multi", "outcomes": outcomes, "groups": outcomes[0]["groups"]}

    # labelled first column + numeric rest as wide treatments? or each col a group
    if outcome_indices and treat_idx is None:
        # A generic first-column factor (e.g. Machine,Strength) is still a factor.
        first_norm = norm[0] if norm else ""
        if first_norm not in ID_HEADERS and all(
            r and parse_numbers(r[0]) == [] for r in rows if r
        ):
            buckets_by_column = []
            for j in outcome_indices:
                buckets: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    if j < len(r):
                        vals = parse_numbers(r[j])
                        if vals and r[0].strip():
                            buckets[collapse_sample_id(r[0])].extend(vals)
                groups = [{"name": k, "values": v} for k, v in buckets.items() if v]
                if len(groups) >= 2:
                    buckets_by_column.append({"parameter": headers[j], "groups": groups})
            if buckets_by_column:
                return {
                    "format": "factor-multi",
                    "outcomes": buckets_by_column,
                    "groups": buckets_by_column[0]["groups"],
                }

        # if first col looks like parameter names and remaining cols are numbers
        first_is_label = all(parse_numbers(r[0]) == [] for r in rows if r)
        if first_is_label and len(headers) >= 3:
            outcomes = []
            for r in rows:
                param = r[0].strip()
                # split remaining numbers into two halves if even, else one list per header
                nums_by_col = []
                for j in range(1, len(r)):
                    nums_by_col.append(parse_numbers(r[j]))
                flat = [n for col in nums_by_col for n in col]
                if len(flat) < 4:
                    continue
                # If headers repeat Brining / Quick
                buckets = defaultdict(list)
                for j in range(1, min(len(headers), len(r))):
                    key = collapse_sample_id(headers[j])
                    if _norm(key) in ID_HEADERS or key.lower().startswith("unnamed") or key.startswith("__"):
                        continue
                    buckets[key].extend(parse_numbers(r[j]))
                groups = [{"name": k, "values": v} for k, v in buckets.items() if v]
                if len(groups) >= 2:
                    outcomes.append({"parameter": param, "groups": groups})
            if outcomes:
                return {"format": "wide-multi", "outcomes": outcomes, "groups": outcomes[0]["groups"]}

        groups = []
        for j in outcome_indices:
            vals = []
            for r in rows:
                if j < len(r):
                    vals.extend(parse_numbers(r[j]))
            if vals:
                groups.append({"name": headers[j], "values": vals})
        if len(groups) >= 2:
            return {"format": "wide", "groups": groups}

    if len(num_idx) == 2 and not factor_indices:
        groups = []
        for j in num_idx:
            groups.append({"name": headers[j], "values": [parse_numbers(r[j])[0] for r in rows if j < len(r) and parse_numbers(r[j])]})
        return {"format": "wide", "groups": groups}
    raise ValueError("Table found but no usable factor or two numeric columns.")


def _has_two_levels(rows: list[list[str]], index: int) -> bool:
    values = {r[index].strip() for r in rows if index < len(r) and r[index].strip()}
    return len(values) >= 2


def _is_constant_column(rows: list[list[str]], index: int) -> bool:
    values = [parse_numbers(r[index])[0] for r in rows if index < len(r) and parse_numbers(r[index])]
    return len(values) > 1 and len(set(values)) == 1


def _composition_only_indices(rows: list[list[str]], headers: list[str], numeric_indices: list[int]) -> set[int]:
    """Identify percent/proportion columns that form a row-wise 100% composition."""
    percent_indices = {
        index for index in numeric_indices
        if "%" in str(headers[index]) or any(token in _norm(headers[index]) for token in ("percent", "proportion", "share"))
    }
    if len(percent_indices) < 2:
        return set()
    complete_rows = []
    for row in rows:
        values = [parse_numbers(row[index])[0] for index in percent_indices if index < len(row) and parse_numbers(row[index])]
        if len(values) == len(percent_indices):
            complete_rows.append(values)
    if complete_rows and all(abs(sum(values) - 100.0) <= 0.01 for values in complete_rows):
        return percent_indices
    return set()


def _looks_like_before_after(name: str) -> bool:
    normalized = _norm(name)
    return any(token in normalized for token in ("before", "after", "pre", "post", "baseline", "followup", "time1", "time2"))


def _detect_paired(
    rows: list[list[str]], headers: list[str], norm: list[str], num_idx: list[int], id_indices: list[int]
) -> dict | None:
    if len(id_indices) != 1:
        return None
    id_idx = id_indices[0]
    named_pairs = len(num_idx) == 2 and all(_looks_like_before_after(headers[i]) for i in num_idx)
    if named_pairs:
        first, second = num_idx
        matched = {}
        for row in rows:
            if max(first, second, id_idx) >= len(row):
                continue
            values_a, values_b = parse_numbers(row[first]), parse_numbers(row[second])
            if values_a and values_b and row[id_idx].strip():
                matched[str(row[id_idx]).strip()] = (values_a[0], values_b[0])
        if len(matched) >= 2:
            return {
                "format": "paired",
                "id": headers[id_idx],
                "before": headers[first],
                "after": headers[second],
                "pairs": [{"id": key, "before": value[0], "after": value[1]} for key, value in matched.items()],
            }

    time_idx = next((i for i, h in enumerate(norm) if h in TIME_HEADERS), None)
    time_outcome_indices = [index for index in num_idx if index != time_idx]
    if time_idx is not None and len(time_outcome_indices) == 1:
        levels = {}
        outcome_idx = time_outcome_indices[0]
        for row in rows:
            if max(id_idx, time_idx, outcome_idx) >= len(row):
                continue
            value = parse_numbers(row[outcome_idx])
            if value and row[id_idx].strip() and row[time_idx].strip():
                levels.setdefault(str(row[id_idx]).strip(), {})[str(row[time_idx]).strip().lower()] = value[0]
        eligible = [
            (key, values) for key, values in levels.items()
            if len(values) >= 2
        ]
        if len(eligible) >= 2:
            labels = list(eligible[0][1])
            named_before = [label for label in labels if "before" in label or "pre" in label or "time1" in label]
            named_after = [label for label in labels if "after" in label or "post" in label or "time2" in label]
            if named_before and named_after:
                before_label, after_label = named_before[0], named_after[0]
            elif len(labels) == 2:
                before_label, after_label = sorted(labels)
            else:
                return {
                    "format": "needs-clarification",
                    "question": "I found repeated ids and a time column, but I cannot identify the before and after levels. Which time level is before?",
                }
            return {
                "format": "paired",
                "id": headers[id_idx],
                "before": before_label,
                "after": after_label,
                "pairs": [{"id": key, "before": values[before_label], "after": values[after_label]} for key, values in eligible],
            }
    if len(num_idx) == 2:
        return {
            "format": "needs-clarification",
            "question": "I found an id with two numeric columns, but I cannot tell whether they are paired before/after values. Are these matched measurements?",
        }
    return None


def _column_numeric(rows: list[list[str]], j: int) -> bool:
    hits, n = 0, 0
    for r in rows:
        if j >= len(r) or not str(r[j]).strip():
            continue
        n += 1
        if parse_numbers(r[j]):
            hits += 1
    return n > 0 and hits / n >= 0.6


def _labelled_lists(text: str) -> list[dict]:
    groups = []
    for ln in text.splitlines():
        if ":" not in ln:
            continue
        name, rest = ln.split(":", 1)
        name = name.strip()
        if not name or _norm(name) in ID_HEADERS:
            continue
        vals = parse_numbers(rest)
        if vals:
            groups.append({"name": collapse_sample_id(name), "values": vals})
    # merge same names
    merged: dict[str, list[float]] = defaultdict(list)
    order = []
    for g in groups:
        if g["name"] not in merged:
            order.append(g["name"])
        merged[g["name"]].extend(g["values"])
    return [{"name": k, "values": merged[k]} for k in order]
