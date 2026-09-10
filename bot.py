"""Rowfirst Telegram bot.

SciPy owns all statistics. Local CSV/XLSX/ZIP files never use Gemini.
Gemini is used only for table extraction from photo/PDF/DOCX when configured.
"""
from __future__ import annotations

import json
import csv
import hashlib
import io
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask

from chapter4 import write_docx
from charts import make_charts
from handle import analyze_ingested, build_breakdown, handle_analyze
from ingest import ingest_file, ingest_text
from qa import quality_check

try:
    import telebot
    from telebot import types
except ImportError:
    telebot = None
    types = None


health_app = Flask(__name__)


@health_app.get("/")
def health_check() -> tuple[str, int]:
    return "ROWFIRST engine active", 200


def _run_health_server() -> None:
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, use_reloader=False)


_TABLE_FACTOR_NAMES = {
    "arm",
    "category",
    "class",
    "condition",
    "group",
    "method",
    "school",
    "sex",
    "site",
    "studytime",
    "treatment",
    "type",
    "variant",
}
_TABLE_ID_NAMES = {
    "code",
    "id",
    "index",
    "patient",
    "patientid",
    "record",
    "recordid",
    "sample",
    "sampleid",
    "student",
    "studentid",
    "subject",
    "subjectid",
    "userid",
}


def _table_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_delimited_frame(raw_text: str) -> pd.DataFrame | None:
    """Read pasted or uploaded delimited text without changing engine logic."""
    raw = (raw_text or "").strip()
    if not raw or "\n" not in raw:
        return None
    try:
        sample = raw[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        separator = dialect.delimiter
    except csv.Error:
        separator = None
    try:
        frame = pd.read_csv(io.StringIO(raw), sep=separator, engine="python")
    except (pd.errors.ParserError, ValueError):
        return None
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return frame if frame.shape[1] >= 2 and not frame.empty else None


def _numeric_table_columns(frame: pd.DataFrame) -> list[str]:
    numeric: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() >= max(1, int(len(frame) * 0.8)):
            numeric.append(column)
    return numeric


def _multivariate_table_route(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Classify a table before it enters the existing ingestion/engine path."""
    if frame is None or frame.shape[1] < 2:
        return None
    numeric_columns = _numeric_table_columns(frame)
    numeric_set = set(numeric_columns)
    identifier_columns = [
        column for column in frame.columns
        if _table_column_name(column) in _TABLE_ID_NAMES
    ]
    factor_columns: list[str] = [
        column for column in frame.columns
        if column not in numeric_set and column not in identifier_columns
    ]
    for column in numeric_columns:
        normalized = _table_column_name(column)
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        integer_like = bool(values.size) and bool((values % 1 == 0).all())
        low_cardinality = values.nunique() >= 2 and values.nunique() <= min(12, max(3, len(frame) // 3))
        if normalized in _TABLE_FACTOR_NAMES or (integer_like and low_cardinality and normalized.endswith("time")):
            factor_columns.append(column)

    # A conventional two-column Treatment/Value table already has an
    # existing ingestion path and should not be diverted to a prompt.
    normalized_headers = {_table_column_name(column) for column in frame.columns}
    if (
        len(frame.columns) == 2
        and normalized_headers & {"group", "method", "treatment", "condition"}
        and len(numeric_columns) == 1
    ):
        return None

    outcome_columns = [column for column in numeric_columns if column not in factor_columns]
    if factor_columns and outcome_columns:
        return {
            "kind": "multivariate",
            "frame": frame,
            "factors": list(dict.fromkeys(factor_columns)),
            "outcomes": outcome_columns,
        }

    if len(numeric_columns) >= 2 and not factor_columns:
        counts = [int(pd.to_numeric(frame[column], errors="coerce").notna().sum()) for column in numeric_columns]
        if any(count < 2 for count in counts):
            return {
                "kind": "invalid",
                "message": (
                    "I found numeric groups with only one observation. "
                    "ANOVA needs at least two observations per group, so I stopped without generating a report."
                ),
            }
        return {"kind": "wide", "frame": frame}
    return None


def _group_size_error(ingested: dict[str, Any]) -> str | None:
    """Reject n=1 groups before the statistical engine is called."""
    groups: list[dict[str, Any]] = list(ingested.get("groups") or [])
    for outcome in ingested.get("outcomes") or []:
        groups.extend(outcome.get("groups") or [])
    if groups and any(len(group.get("values") or []) < 2 for group in groups):
        return (
            "I found a group with only one observation. "
            "ANOVA needs at least two observations per group, so I stopped without generating a report."
        )
    return None


def _analyze_ingested_safely(ingested: dict[str, Any], outcome_name: str | None = None) -> dict[str, Any]:
    guard_error = _group_size_error(ingested)
    if guard_error:
        return {"ok": False, "error": guard_error}
    return analyze_ingested(ingested, outcome_name=outcome_name)


def _multivariate_file_route(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        return None
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    return _multivariate_table_route(_read_delimited_frame(raw))


def _group_multivariate_frame(
    frame: pd.DataFrame,
    factor: str,
    outcome: str,
) -> list[dict[str, Any]]:
    selected = frame[[factor, outcome]].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    selected = selected.dropna(subset=[factor, outcome])
    grouped = selected.groupby(factor, dropna=False)[outcome].apply(list)
    return [
        {"name": str(name), "values": [float(value) for value in values]}
        for name, values in grouped.items()
        if values
    ]


def explain(engine: dict[str, Any]) -> str:
    """Only display verified engine text; do not ask an LLM to rewrite statistics."""
    if not engine.get("ok"):
        return engine.get("error", "I could not analyse that.")
    return engine["message"]


def _table_preview(table: dict[str, Any]) -> str:
    lines = ["Extracted table:"]
    lines.append(" | ".join(str(header) for header in table["headers"]))
    lines.append("-" * min(120, max(3, len(lines[-1]))))
    for row in table["rows"]:
        lines.append(" | ".join(str(value) for value in row))
    lines.append("Reply YES to analyse this table.")
    return "\n".join(lines)


def _engine_from_file(path: Path) -> dict[str, Any]:
    ingested = ingest_file(path)
    engine = _analyze_ingested_safely(ingested)
    engine["ingested"] = ingested
    engine["qa"] = quality_check(ingested)
    engine["breakdown"] = build_breakdown(engine)
    return engine


LOCAL_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".zip"}
GEMINI_SUFFIXES = {".pdf", ".docx", ".doc"}
GEMINI_FALLBACK = (
    "Add GEMINI_API_KEY in Secrets to read PDF/Word. You can still paste the table or send CSV."
)
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_UNAVAILABLE = (
    "Gemini could not read this file because the configured model is unavailable. "
    "Please try again later or paste the table/send CSV."
)
GEMINI_FAILED = (
    "Gemini could not read this file right now. Please try again or paste the table/send CSV."
)
MIME_SUFFIXES = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/plain": ".txt",
    "text/tab-separated-values": ".tsv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
}


class GeminiExtractionError(RuntimeError):
    """An extraction failure whose safe message can be shown to users."""


class GeminiNoTableError(GeminiExtractionError):
    """Gemini returned no usable table."""


def _upload_suffix(filename: str, mime_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in LOCAL_SUFFIXES or suffix in GEMINI_SUFFIXES:
        return suffix
    return MIME_SUFFIXES.get((mime_type or "").split(";", 1)[0].strip().lower(), "")


def _gemini_extract(path: Path) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise GeminiExtractionError(GEMINI_FALLBACK)
    try:
        import google.generativeai as genai

        # gemini-2.0-flash is no longer available for this API key. Keep the
        # model explicit so a retired model does not look like a missing key.
        mime = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = (
            'Extract only tables as JSON {"headers": [...], "rows": [...]}. '
            "No statistics. No chapter. Preserve the headers, labels, and values exactly."
        )
        response = model.generate_content([prompt, {"mime_type": mime, "data": path.read_bytes()}])
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
        payload = json.loads(raw)
        headers = payload.get("headers")
        rows = payload.get("rows")
        if not headers or not isinstance(headers, list) or not isinstance(rows, list) or not rows:
            raise GeminiNoTableError("I couldn’t find a table in that file. Please send a file with a table or paste it.")
        if any(not isinstance(row, list) for row in rows):
            raise GeminiNoTableError("I couldn’t find a table in that file. Please send a file with a table or paste it.")
        return {"headers": headers, "rows": rows}
    except GeminiExtractionError:
        raise
    except Exception as exc:
        print(f"Gemini extraction failed: {type(exc).__name__}", flush=True)
        if type(exc).__name__ == "NotFound":
            raise GeminiExtractionError(GEMINI_UNAVAILABLE) from exc
        raise GeminiExtractionError(GEMINI_FAILED) from exc


def _table_as_csv(table: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(table["headers"])
    writer.writerows(table["rows"])
    return output.getvalue()


def _send_analysis(
    bot: Any,
    message: Any,
    engine: dict[str, Any],
    *,
    ask_for_document: bool = True,
) -> None:
    if not engine.get("ok"):
        bot.reply_to(message, engine.get("question") or engine.get("error", "I could not analyse that.")[:4000])
        return
    results = engine.get("results") or [engine.get("result")]
    results = [result for result in results if result]
    result_lines = ["Outcome | Test | Statistic | p | Decision", *[_summary_line(result) for result in results]]
    breakdown = engine.get("breakdown") or build_breakdown(engine)
    qa = engine.get("qa") or {}
    qa_lines = qa.get("warnings", []) + qa.get("errors", [])
    qa_text = "\n".join(f"- {item}" for item in qa_lines) if qa_lines else "none"
    if "charts" not in engine:
        try:
            engine["charts"] = make_charts(
                engine,
                tempfile.mkdtemp(prefix="rowfirst-charts-"),
            )
        except Exception as exc:
            print(f"Chart generation skipped: {type(exc).__name__}", flush=True)
            engine["charts"] = []
    _reply_block(bot, message, "RESULTS\n" + "\n".join(result_lines))
    _reply_block(bot, message, "BREAKDOWN\n" + breakdown)
    _reply_block(bot, message, "QA\n" + qa_text)
    for chart in engine.get("charts", [])[:6]:
        chart_path = Path(chart["path"])
        if not chart_path.exists():
            continue
        try:
            with chart_path.open("rb") as image_file:
                bot.send_photo(message.chat.id, image_file, caption=chart.get("caption", ""))
        except Exception as exc:
            print(f"Chart delivery skipped: {type(exc).__name__}", flush=True)
    if ask_for_document:
        bot.reply_to(message, "Want a Results Document (Word)? Reply YES")


def _reply_block(bot: Any, message: Any, text: str) -> None:
    if len(text) > 3900:
        text = text[:3890].rstrip() + "\n…"
    bot.reply_to(message, text)


def _send_error(bot: Any, message: Any, exc: Exception, prefix: str = "Could not read your file") -> None:
    detail = str(exc).strip() or exc.__class__.__name__
    text = detail if isinstance(exc, GeminiExtractionError) else f"{prefix}: {detail}"
    try:
        bot.reply_to(message, text[:4000])
    except Exception as reply_exc:
        print(f"{text}; Telegram error while sending it: {reply_exc}")


def _is_small_talk(text: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", (text or "").lower()).strip()
    return cleaned in {"hi", "hello", "hey", "thanks", "thank you", "thx"}


def _analysis_request(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    topic_match = re.search(r"(?im)^\s*topic\s*:\s*(.+?)\s*$", text or "")
    outcome_match = re.search(
        r"(?im)^\s*(?:outcome|dependent\s+variable|dependent_variable|response|measure|metric)\s*:\s*(.+?)\s*$",
        text or "",
    )
    hypothesis_matches = re.finditer(
        r"(?im)^\s*(h(?:0|o)\d*)\s*:\s*(.+?)\s*$",
        text or "",
    )
    hypotheses = [
        {"label": match.group(1), "text": match.group(2).strip()}
        for match in hypothesis_matches
    ]
    data_lines = []
    for line in (text or "").splitlines():
        if re.match(r"(?i)^\s*(topic|outcome|dependent\s+variable|dependent_variable|response|measure|metric|h(?:0|o)\d*)\s*:", line):
            continue
        data_lines.append(line)
    metadata: dict[str, Any] = {
        "topic": topic_match.group(1).strip() if topic_match else "",
        "hypotheses": hypotheses,
        "discuss": bool(re.search(r"\bdiscuss\b", text or "", flags=re.I)),
    }
    request = {"text": "\n".join(data_lines).strip()}
    if outcome_match:
        request["outcome"] = outcome_match.group(1).strip()
    return request, metadata


def _run_analysis(text: str) -> dict[str, Any]:
    request, metadata = _analysis_request(text)
    try:
        ingested = ingest_text(request["text"])
    except (ValueError, TypeError):
        ingested = None
    if ingested:
        guard_error = _group_size_error(ingested)
        if guard_error:
            return {"ok": False, "error": guard_error}
    engine = handle_analyze(request)
    if engine.get("ok"):
        if metadata["topic"]:
            engine["topic"] = metadata["topic"]
        if metadata["hypotheses"]:
            engine["hypotheses"] = metadata["hypotheses"]
        if metadata["discuss"]:
            engine["discuss"] = True
    return engine


def _defense_qa(engine: dict[str, Any]) -> str:
    results = engine.get("results") or [engine.get("result")]
    results = [result for result in results if result]
    statistics = []
    p_values = []
    for result in results:
        test = result.get("test")
        if test in {"student-t", "welch-t", "paired-t"}:
            statistics.append(f"t={result['t']:.12g}")
            p_values.append(f"p={result['p']:.12g}")
        elif test == "one-way anova":
            statistics.append(f"F={result['F']:.12g}")
            p_values.append(f"p={result['p']:.12g}")
        elif test == "two-way anova":
            for effect in result.get("effects", []):
                statistics.append(f"F={effect['F']:.12g}")
                p_values.append(f"p={effect['p']:.12g}")
        elif "p" in result:
            p_values.append(f"p={result['p']:.12g}")
    stat_text = ", ".join(statistics) or "No t or F statistic was reported by the engine."
    p_text = ", ".join(p_values) or "No p-value was reported by the engine."
    decision = "significant at α = .05" if any(result.get("isSignificant") for result in results) else "not significant at α = .05"
    return "\n".join([
        "Defense Q&A",
        f"1. What statistic did the engine report?\n{stat_text}.",
        f"2. What exact p-value did it report?\n{p_text}.",
        f"3. Was the result significant at 5%?\n{decision}; {p_text}.",
        f"4. What evidence should be quoted?\n{stat_text}; {p_text}.",
        f"5. What does the result support?\nThe engine supports only the {stat_text} and {p_text} reported above.",
    ])


def _acquire_polling_lock(token: str):
    """Allow only one bot process to poll a given Telegram token."""
    import fcntl

    lock_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f"rowfirst-telegram-{lock_id}.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def _summary_line(result: dict[str, Any]) -> str:
    test = result.get("test", "")
    outcome = result.get("parameter") or result.get("outcome") or "Measured outcome"
    if test == "one-way anova":
        statistic = f"F({result['dfb']},{result['dfw']})={result['F']:.4f}"
    elif test in {"student-t", "welch-t", "paired-t"}:
        statistic = f"t({result['df']:.0f})={result['t']:.4f}"
    elif test == "simple linear regression":
        statistic = f"r={result['r']:.4f}; r²={result['rSquared']:.4f}"
    elif test == "two-way anova":
        statistic = "; ".join(
            f"{effect['effect']} F={effect['F']:.4f}" for effect in result.get("effects", [])
        ) or "multiple effects"
    elif test in {"pearson", "spearman"}:
        statistic = f"r={result['r']:.4f}"
    elif test == "fisher-exact":
        statistic = f"Fisher exact; OR={result['oddsRatio']:.4f}"
    elif test == "chi-square":
        statistic = f"χ²({result['df']})={result['chi2']:.4f}"
    else:
        statistic = "not reported"
    if test == "two-way anova":
        p = "; ".join(
            f"{effect['effect']} {_p(effect['p'])}" for effect in result.get("effects", [])
        ) or "not reported"
        decision = ", ".join(
            effect["effect"] for effect in result.get("effects", []) if effect.get("isSignificant")
        ) or "not significant"
    else:
        p = _p(result["p"]) if "p" in result else "not reported"
        decision = "significant" if result.get("isSignificant") else "not significant"
    return f"{outcome} | {test} | {statistic} | {p} | {decision}"


def _p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not reported"
    return "< .001" if number < 0.001 else f"{number:.4f}"


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in Secrets.")
    if telebot is None:
        raise SystemExit("Install pyTelegramBotAPI from requirements.txt.")
    polling_lock = _acquire_polling_lock(token)
    if polling_lock is None:
        print("Telegram bot already polling this token; exiting.", flush=True)
        return
    bot = telebot.TeleBot(token)
    pending: dict[int, dict[str, str]] = {}
    pending_multivariate: dict[int, dict[str, Any]] = {}
    user_sessions: dict[int, dict[str, Any]] = {}
    last_engine: dict[int, dict[str, Any]] = {}
    text_buffers: dict[int, dict[str, Any]] = {}
    text_buffer_lock = threading.Lock()
    text_debounce_seconds = 1.5
    text_chunk_cooldown_seconds = 2.0

    @bot.message_handler(commands=["start", "help"])
    def start(message: Any) -> None:
        try:
            bot.reply_to(
                message,
                "Hey — I’m Rowfirst. Send a table (paste, csv, excel, photo) and I’ll run the test.\n"
                "I’ll ask if the groups aren’t clear.",
            )
        except Exception as exc:
            _send_error(bot, message, exc, "Could not send the welcome message")

    def _session_actions_markup() -> Any:
        markup = types.InlineKeyboardMarkup(row_width=1)
        markup.add(
            types.InlineKeyboardButton("🔄 Test Another Variable", callback_data="session:rerun"),
            types.InlineKeyboardButton("📁 Clear & New Dataset", callback_data="session:clear"),
        )
        return markup

    def send_results_document(message: Any) -> bool:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return False
        try:
            with tempfile.TemporaryDirectory(prefix="rowfirst-results-") as tmp:
                docx_path = Path(tmp) / "Rowfirst_Results.docx"
                write_docx(engine, docx_path)
                with open(docx_path, "rb") as document_file:
                    bot.send_document(
                        message.chat.id,
                        document_file,
                        caption="Rowfirst_Results.docx — built from the last engine JSON.",
                    )
            if message.chat.id in user_sessions:
                bot.reply_to(
                    message,
                    "Analysis complete! What would you like to do next?",
                    reply_markup=_session_actions_markup(),
                )
            return True
        except Exception as exc:
            _send_error(bot, message, exc, "Could not build the Results Document")
            return False

    def send_defense(message: Any) -> None:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return
        _reply_block(bot, message, _defense_qa(engine))

    @bot.message_handler(commands=["results"])
    def results_document(message: Any) -> None:
        send_results_document(message)

    @bot.message_handler(commands=["full"])
    def full(message: Any) -> None:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return
        _reply_block(
            bot,
            message,
            "RESULTS\n"
            + "\n".join(
                ["Outcome | Test | Statistic | p | Decision"]
                + [_summary_line(result) for result in (engine.get("results") or [engine.get("result")]) if result]
            ),
        )

    def _choice_markup(prefix: str, choices: list[str]) -> Any:
        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(*[
            types.InlineKeyboardButton(
                text=str(choice)[:50],
                callback_data=f"{prefix}:{choice}",
            )
            for choice in choices
        ])
        return markup

    def _prompt_multivariate(
        message: Any,
        route: dict[str, Any],
        *,
        cache_session: bool = True,
    ) -> None:
        if cache_session:
            user_sessions[message.chat.id] = {
                "df": route["frame"],
                "timestamp": time.time(),
            }
        pending_multivariate[message.chat.id] = {
            "frame": route["frame"],
            "factors": route["factors"],
            "outcomes": route["outcomes"],
        }
        bot.reply_to(
            message,
            "Step 1/2: Select the Grouping Factor (Independent Variable)",
            reply_markup=_choice_markup("factor", route["factors"]),
        )

    @bot.callback_query_handler(
        func=lambda call: (getattr(call, "data", "") or "").startswith("session:")
    )
    def on_session_selection(call: Any) -> None:
        chat_id = call.message.chat.id
        callback_data = call.data or ""
        bot.answer_callback_query(call.id)
        if callback_data == "session:rerun":
            session = user_sessions.get(chat_id)
            if not session:
                bot.reply_to(call.message, "Session expired. Please paste or upload your dataset again.")
                return
            route = _multivariate_table_route(session.get("df"))
            if not route or route.get("kind") != "multivariate":
                bot.reply_to(call.message, "Session expired. Please paste or upload your dataset again.")
                return
            _prompt_multivariate(call.message, route, cache_session=False)
            return
        if callback_data == "session:clear":
            user_sessions.pop(chat_id, None)
            pending_multivariate.pop(chat_id, None)
            bot.reply_to(call.message, "Session cleared. Ready for your next dataset!")
            return
        bot.answer_callback_query(call.id, "Invalid session action.", show_alert=True)

    def _maybe_route_multivariate(message: Any, raw_text: str) -> bool:
        route = _multivariate_table_route(_read_delimited_frame(raw_text))
        if not route:
            return False
        if route["kind"] == "invalid":
            bot.reply_to(message, route["message"])
            return True
        if route["kind"] == "multivariate":
            _prompt_multivariate(message, route)
            return True
        return False

    def _has_header_line(raw_text: str) -> bool:
        first_line = next((line.strip() for line in raw_text.splitlines() if line.strip()), "")
        if not first_line:
            return False
        try:
            dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t")
            cells = next(csv.reader([first_line], dialect))
        except (csv.Error, StopIteration):
            return False
        return len(cells) >= 2 and any(re.search(r"[A-Za-z]", cell or "") for cell in cells)

    def _looks_like_pasted_table(raw_text: str) -> bool:
        lines = [line for line in raw_text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        first_line = lines[0]
        return any(delimiter in first_line for delimiter in (",", ";", "\t"))

    def _process_pasted_text(message: Any, accumulated_text: str) -> None:
        chat_id = message.chat.id
        line_count = len([line for line in accumulated_text.splitlines() if line.strip()])
        if line_count > 50:
            bot.reply_to(
                message,
                "Tip: For large datasets (100+ rows), uploading as a .csv or .txt file avoids Telegram message splitting.",
            )
        if chat_id in pending_multivariate:
            bot.reply_to(message, "Please finish the current column selection before sending another dataset.")
            return
        if _maybe_route_multivariate(message, accumulated_text):
            return
        engine = _run_analysis(accumulated_text)
        if engine.get("ok"):
            last_engine[chat_id] = engine
        _send_analysis(bot, message, engine)

    def _flush_text_buffer(chat_id: int) -> None:
        with text_buffer_lock:
            state = text_buffers.pop(chat_id, None)
        if not state:
            return
        try:
            _process_pasted_text(state["message"], state["text"])
        except Exception as exc:
            _send_error(bot, state["message"], exc, "Could not analyse that")

    def _queue_pasted_text(message: Any, text: str) -> bool:
        if not _looks_like_pasted_table(text):
            return False
        chat_id = message.chat.id
        now = time.monotonic()
        has_header = _has_header_line(text)
        with text_buffer_lock:
            previous = text_buffers.get(chat_id)
            within_cooldown = bool(
                previous and now - previous["received_at"] <= text_chunk_cooldown_seconds
            )
            if previous and within_cooldown:
                accumulated = previous["text"].rstrip() + "\n" + text.lstrip()
                timer = previous.get("timer")
                if timer:
                    timer.cancel()
            else:
                accumulated = text
            timer = threading.Timer(
                text_debounce_seconds,
                _flush_text_buffer,
                args=(chat_id,),
            )
            timer.daemon = True
            text_buffers[chat_id] = {
                "text": accumulated,
                "message": message,
                "received_at": now,
                "has_header": bool(previous and within_cooldown and (previous["has_header"] or has_header)),
                "timer": timer,
            }
            timer.start()
        return True

    @bot.callback_query_handler(func=lambda call: (getattr(call, "data", "") or "").startswith(("factor:", "outcome:")))
    def on_multivariate_selection(call: Any) -> None:
        chat_id = call.message.chat.id
        state = pending_multivariate.get(chat_id)
        if not state:
            bot.answer_callback_query(call.id, "This dataset selection has expired.", show_alert=True)
            return
        bot.answer_callback_query(call.id)
        callback_data = call.data or ""
        if ":" not in callback_data:
            bot.answer_callback_query(call.id, "Invalid selection.", show_alert=True)
            return
        selection_type, selected_column = callback_data.split(":", 1)
        if selection_type not in {"factor", "outcome"} or not selected_column:
            bot.answer_callback_query(call.id, "Invalid selection.", show_alert=True)
            return

        if selection_type == "factor":
            factors = state["factors"]
            if selected_column not in factors:
                bot.answer_callback_query(call.id, "Invalid grouping factor.", show_alert=True)
                return
            state["chosen_factor"] = selected_column
            bot.edit_message_text(
                "Step 2/2: Select the Outcome Variable to measure (Dependent Variable)",
                chat_id=chat_id,
                message_id=call.message.message_id,
                reply_markup=_choice_markup("outcome", state["outcomes"]),
            )
            return

        outcomes = state["outcomes"]
        chosen_factor = state.get("chosen_factor")
        if selected_column not in outcomes or not chosen_factor:
            bot.answer_callback_query(call.id, "Select a grouping factor first.", show_alert=True)
            return
        state["chosen_outcome"] = selected_column
        chosen_outcome = state["chosen_outcome"]
        pending_multivariate.pop(chat_id, None)
        frame = state["frame"]
        groups = [
            group.dropna().values
            for _, group in frame.groupby(chosen_factor)[chosen_outcome]
        ]
        if any(len(group) < 2 for group in groups):
            bot.reply_to(
                call.message,
                "I found a group with only one observation. "
                "ANOVA needs at least two observations per group, so I stopped without generating a report.",
            )
            return
        grouped = frame.groupby(chosen_factor)[chosen_outcome]
        ingested = {
            "format": "labelled",
            "groups": [
                {
                    "name": str(group_name),
                    "values": [float(value) for value in group.dropna().values],
                }
                for group_name, group in grouped
            ],
        }
        engine = _analyze_ingested_safely(ingested, outcome_name=str(chosen_outcome))
        engine["ingested"] = ingested
        engine["qa"] = quality_check(ingested)
        engine["breakdown"] = build_breakdown(engine)
        if engine.get("ok"):
            last_engine[chat_id] = engine
            _send_analysis(bot, call.message, engine, ask_for_document=False)
            send_results_document(call.message)
        else:
            _send_analysis(bot, call.message, engine)

    @bot.message_handler(content_types=["text"])
    def on_text(message: Any) -> None:
        try:
            text = message.text or ""
            if text.strip().startswith("/"):
                return
            if _is_small_talk(text):
                bot.reply_to(message, "Hey — good to hear from you.\nSend a table, CSV, Excel file, or photo and I’ll help run the test.")
                return
            if re.search(r"\b(defense|viva)\b", text, flags=re.I):
                send_defense(message)
                return
            if message.chat.id in pending_multivariate:
                bot.reply_to(message, "Please finish the current column selection before sending another dataset.")
                return
            if (
                text.strip().upper() in {"YES", "YES4"}
                and message.chat.id in last_engine
                and message.chat.id not in pending
            ):
                send_results_document(message)
                return
            if text.strip().upper() == "YES" and message.chat.id in pending:
                analysis_text = pending.pop(message.chat.id)["text"]
                if _maybe_route_multivariate(message, analysis_text):
                    return
                engine = _run_analysis(analysis_text)
            elif message.chat.id in pending:
                bot.reply_to(message, "I have the extracted table ready. Reply YES to analyse it.")
                return
            else:
                if _queue_pasted_text(message, text):
                    return
                _process_pasted_text(message, text)
                return
            if engine.get("ok"):
                last_engine[message.chat.id] = engine
            _send_analysis(bot, message, engine)
        except Exception as exc:
            _send_error(bot, message, exc, "Could not analyse that")

    @bot.message_handler(content_types=["document"])
    def on_document(message: Any) -> None:
        try:
            bot.reply_to(message, "Got it, reading your file…")
            document = message.document
            name = getattr(document, "file_name", None) or "upload"
            mime_type = getattr(document, "mime_type", "") or ""
            suffix = _upload_suffix(name, mime_type)
            if not suffix:
                raise ValueError("Please send a CSV, TXT, TSV, Excel, ZIP, PDF, or DOCX file.")
            with tempfile.TemporaryDirectory(prefix="rowfirst-upload-") as tmp:
                safe_name = Path(name).name
                if Path(safe_name).suffix.lower() != suffix:
                    safe_name = f"{safe_name}{suffix}"
                path = Path(tmp) / safe_name
                info = bot.get_file(document.file_id)
                path.write_bytes(bot.download_file(info.file_path))
                if suffix in LOCAL_SUFFIXES:
                    route = _multivariate_file_route(path)
                    if route and route["kind"] == "invalid":
                        bot.reply_to(message, route["message"])
                        return
                    if route and route["kind"] == "multivariate":
                        _prompt_multivariate(message, route)
                        return
                    engine = _engine_from_file(path)
                    if engine.get("ok"):
                        last_engine[message.chat.id] = engine
                    _send_analysis(bot, message, engine)
                    return
                if suffix in GEMINI_SUFFIXES:
                    table = _gemini_extract(path)
                    pending[message.chat.id] = {
                        "text": _table_as_csv(table),
                        "source": suffix,
                    }
                    _reply_block(bot, message, _table_preview(table))
                    return
        except Exception as exc:
            _send_error(bot, message, exc)

    @bot.message_handler(content_types=["photo"])
    def on_photo(message: Any) -> None:
        try:
            bot.reply_to(message, "Got it, reading your file…")
            with tempfile.TemporaryDirectory(prefix="rowfirst-photo-") as tmp:
                path = Path(tmp) / "photo.jpg"
                info = bot.get_file(message.photo[-1].file_id)
                path.write_bytes(bot.download_file(info.file_path))
                table_text = _gemini_extract(path)
                engine = handle_analyze({"text": _table_as_csv(table_text)})
                if engine.get("ok"):
                    last_engine[message.chat.id] = engine
                _send_analysis(bot, message, engine)
        except Exception as exc:
            _send_error(bot, message, exc, "Could not read the photo")

    threading.Thread(
        target=_run_health_server,
        name="rowfirst-health-server",
        daemon=True,
    ).start()
    print("polling started once", flush=True)
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()