# Rowfirst Analytics Bot

Telegram-based research analysis bot that runs verified statistical tests, creates charts and result documents, and writes plain-English university-style discussions.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `python3 tests/test_gold.py` — verify the SciPy gold outputs and reporting resilience
- `python3 bot.py` — start Telegram polling; requires `TELEGRAM_BOT_TOKEN` and `GEMINI_API_KEY` in Secrets

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Python 3.11, SciPy, statsmodels, pandas, pyTelegramBotAPI, and reportlab
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `bot.py` — Telegram handlers and document delivery
- `handle.py` — analysis orchestration, verified result formatting, and Telegram breakdown
- `stats_engine.py` — deterministic statistical calculations owned by SciPy/statsmodels
- `chapter4.py` — Markdown, DOCX, and PDF result documents, including Section 7 discussion
- `ingest.py` — CSV, spreadsheet, text, ZIP, and paired/two-way input handling
- `tests/test_gold.py` — A–F gold tests and reporting checks

## Architecture decisions

- Statistical calculations stay deterministic in `stats_engine.py`; narrative code does not recompute results.
- Section 7 synthesizes all outcomes in plain English instead of repeating Section 6.
- Design-aware wording is conditional: ANOVA, moisture/drier, pH acidity, and other language only appear when supported by the analyzed columns and test.

## Product

Researchers can send tables and supported files to the Telegram bot, receive verified statistical results, readable breakdowns, charts, and a downloadable Results Document.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Only one Telegram polling process may run for the bot token at a time.
- Never hardcode or print `TELEGRAM_BOT_TOKEN` or `GEMINI_API_KEY`.
- Do not rewrite SciPy/statistical engine logic while changing reporting language.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
