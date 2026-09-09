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
from pathlib import Path
from typing import Any

from chapter4 import write_docx
from charts import make_charts
from handle import analyze_ingested, build_breakdown, handle_analyze
from ingest import ingest_file
from qa import quality_check

try:
    import telebot
    from telebot import types
except ImportError:
    telebot = None
    types = None


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
    engine = analyze_ingested(ingested)
    engine["ingested"] = ingested
    engine["qa"] = quality_check(ingested)
    engine["breakdown"] = build_breakdown(engine)
    return engine


LOCAL_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".zip"}
GEMINI_SUFFIXES = {".pdf", ".docx", ".doc"}
GEMINI_FALLBACK = (
    "Add GEMINI_API_KEY in Secrets to read PDF/Word. You can still paste the table or send CSV."
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
        model = genai.GenerativeModel("gemini-2.0-flash")
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
        raise GeminiExtractionError(GEMINI_FALLBACK) from exc


def _table_as_csv(table: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(table["headers"])
    writer.writerows(table["rows"])
    return output.getvalue()


def _send_analysis(bot: Any, message: Any, engine: dict[str, Any]) -> None:
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
        if re.match(r"(?i)^\s*(topic|h(?:0|o)\d*)\s*:", line):
            continue
        data_lines.append(line)
    metadata: dict[str, Any] = {
        "topic": topic_match.group(1).strip() if topic_match else "",
        "hypotheses": hypotheses,
        "discuss": bool(re.search(r"\bdiscuss\b", text or "", flags=re.I)),
    }
    return {"text": "\n".join(data_lines).strip()}, metadata


def _run_analysis(text: str) -> dict[str, Any]:
    request, metadata = _analysis_request(text)
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
    last_engine: dict[int, dict[str, Any]] = {}

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

    def send_results_document(message: Any) -> None:
        engine = last_engine.get(message.chat.id)
        if not engine or not engine.get("ok"):
            bot.reply_to(message, "Send a table first.")
            return
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
        except Exception as exc:
            _send_error(bot, message, exc, "Could not build the Results Document")

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
            if (
                text.strip().upper() in {"YES", "YES4"}
                and message.chat.id in last_engine
                and message.chat.id not in pending
            ):
                send_results_document(message)
                return
            if text.strip().upper() == "YES" and message.chat.id in pending:
                analysis_text = pending.pop(message.chat.id)["text"]
                engine = _run_analysis(analysis_text)
            elif message.chat.id in pending:
                bot.reply_to(message, "I have the extracted table ready. Reply YES to analyse it.")
                return
            else:
                engine = _run_analysis(text)
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

    print("polling started once", flush=True)
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()