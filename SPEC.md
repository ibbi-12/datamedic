# DataMedic — Self-Healing Data Analyst Agent

## One-line pitch
Upload any CSV, ask a question in plain English. The agent writes pandas/matplotlib code, runs it in a sandbox, reads its own tracebacks when it fails, and rewrites the code until it works — then returns the chart plus a written insight.

## Why this exists
Portfolio project demonstrating: agentic self-correction loops, sandboxed code execution, structured LLM output, and LangGraph state machines. Built to be demoable in a 30-second GIF.

---

## Tech stack (fixed — do not substitute)
- **Python 3.11+**
- **LangGraph** for the agent graph
- **Groq API** with `llama-3.3-70b-versatile` (fast, free tier) — read key from env var `GROQ_API_KEY`
- **Sandbox:** local `subprocess` running Python in an isolated temp dir with a timeout (no Docker, keep weekend scope). Whitelist imports: pandas, numpy, matplotlib, seaborn.
- **FastAPI** backend
- **Frontend:** single-page vanilla HTML/JS/CSS served by FastAPI (no React build step). Clean, minimal, dark theme.
- **uv** for dependency management (`pyproject.toml`)

---

## Core agent loop (LangGraph)

### State schema
```python
class AgentState(TypedDict):
    csv_path: str
    csv_profile: str          # column names, dtypes, 5 sample rows, null counts
    question: str
    plan: str                 # short natural-language analysis plan
    code: str                 # current code attempt
    stdout: str
    stderr: str
    attempt: int              # current attempt number
    max_attempts: int         # default 4
    chart_path: str | None    # PNG output if produced
    result_summary: str       # final written insight
    history: list[dict]       # every attempt: {code, stderr, critique} — shown in UI
    status: str               # "planning" | "coding" | "executing" | "fixing" | "done" | "failed"
```

### Nodes
1. **profile_csv** — deterministic (no LLM). Load CSV with pandas, produce `csv_profile`: column names, dtypes, shape, null counts, `df.head(5).to_string()`. Truncate to ~1500 chars.
2. **plan** — LLM. Input: profile + question. Output: 2-4 sentence analysis plan (what to compute, what chart type, if any). Store in `plan`.
3. **write_code** — LLM. Input: profile + question + plan + (if retry) previous code + stderr + critique. Output: a complete standalone Python script. Requirements enforced in the prompt:
   - Reads the CSV from the literal path in `csv_path`
   - Prints key findings to stdout
   - If a chart is appropriate, saves exactly one PNG to a literal output path provided in the prompt (`chart_path` in a temp dir); never `plt.show()`
   - Only uses whitelisted libraries
   - Respond ONLY with code in one ```python block; parser strips fences
4. **execute** — deterministic. Write code to a temp file, run `subprocess.run([sys.executable, file], capture_output=True, timeout=30, cwd=tempdir)`. Capture stdout/stderr. Check whether the PNG was created.
5. **critique** — LLM, only on failure. Input: code + stderr. Output: 1-3 sentence diagnosis of the bug and the fix strategy. Append `{code, stderr, critique}` to `history`.
6. **summarize** — LLM, on success. Input: question + plan + code stdout. Output: 3-6 sentence plain-English insight answering the user's question, referencing actual numbers from stdout.

### Edges
- profile_csv → plan → write_code → execute
- execute → summarize (exit code 0 AND stdout non-empty)
- execute → critique (non-zero exit / timeout / empty output)
- critique → write_code (if attempt < max_attempts)
- critique → END with status "failed" (attempts exhausted) — return the full history so the user sees what was tried
- summarize → END with status "done"

### Safety constraints on generated code
Before execution, statically reject code containing: `import os`, `import sys` (except the runner's own use), `subprocess`, `open(` for writing outside the temp dir, `requests`, `urllib`, `socket`, `shutil`, `eval(`, `exec(`, `__import__`. If rejected, treat as a failure with a synthetic stderr message ("blocked import: X") and route to critique. Run the subprocess with `cwd` set to a fresh temp dir per request.

---

## API (FastAPI)

- `POST /analyze` — multipart: `file` (CSV, max 10 MB), `question` (str). Returns a `job_id` immediately and runs the graph in a background task.
- `GET /status/{job_id}` — returns current `status`, `attempt`, and `history` so the UI can live-update ("Attempt 2: fixing KeyError...").
- `GET /result/{job_id}` — returns `result_summary`, final `code`, full `history`, and chart URL if present.
- `GET /chart/{job_id}` — serves the PNG.
- Keep job state in an in-memory dict (fine for demo; note it in README as a known limitation).

## Frontend (single page)
- Drag-and-drop CSV upload + question input + submit button.
- Live status panel that polls `/status`: show each attempt as a card — collapsed code block, red stderr snippet, and the critique line. This visible retry loop IS the demo. Make failures look good.
- On success: chart image, insight paragraph, and expandable final code block with a copy button.
- Dark theme, monospace for code, subtle green/red status accents. No frameworks.

---

## Repo layout
```
datamedic/
├── README.md
├── pyproject.toml
├── .env.example          # GROQ_API_KEY=
├── app/
│   ├── main.py           # FastAPI app + routes + static serving
│   ├── graph.py          # LangGraph build: nodes, edges, state
│   ├── nodes.py          # node implementations
│   ├── sandbox.py        # static checks + subprocess runner
│   ├── prompts.py        # all LLM prompts as constants
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── examples/
│   ├── titanic.csv       # download from a public source or generate synthetic
│   └── sales.csv         # synthetic messy sales data WITH deliberate quirks
│                         #   (mixed date formats, a currency column with "$" strings,
│                         #    nulls) so the agent visibly fails once and self-heals
└── tests/
    └── test_sandbox.py   # static-check rejection cases + happy path
```

## README requirements (write it as part of the build)
- Hero: one-line pitch + demo GIF placeholder (`![demo](docs/demo.gif)`)
- Architecture diagram of the graph (Mermaid)
- "How self-healing works" section showing a real attempt→traceback→critique→fix sequence
- Quickstart: `uv sync`, set env, `uvicorn app.main:app`, open localhost
- Known limitations (in-memory jobs, subprocess sandbox not production-grade, single user)

## Definition of done
1. `uvicorn app.main:app` starts with no errors.
2. Uploading `examples/sales.csv` with "What's the monthly revenue trend?" produces at least one failed attempt (due to the messy currency column), a visible critique, and then a successful chart + summary.
3. Uploading `examples/titanic.csv` with "Did passenger class affect survival?" succeeds and produces a grouped bar chart.
4. A blocked import (e.g. agent generates `import os`) is caught, critiqued, and recovered from.
5. Tests pass: `pytest`.

## Build order (suggested for the agent)
1. sandbox.py + tests
2. prompts.py + nodes.py + graph.py, tested via a CLI script against examples/
3. FastAPI routes with background jobs
4. Frontend
5. README + example CSVs

## Non-goals (do not build)
Auth, multi-user persistence, Docker, database, deployment configs, chart type selection UI, multi-file uploads.
