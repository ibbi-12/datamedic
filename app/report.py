"""Standalone HTML report builder: one self-contained file, charts inlined as base64."""

from __future__ import annotations

import base64
import html
from datetime import date
from pathlib import Path
from typing import Any

_CSS = """
:root { --ground:#15120d; --panel:#1e1912; --border:#362d21; --ink:#ede6d9;
  --muted:#93897a; --accent:#e8a23d; --success:#4cbb82; --danger:#e8604a;
  --mono:"IBM Plex Mono","SFMono-Regular",Menlo,monospace; }
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.6; }
main { max-width:760px; margin:0 auto; padding:3rem 1.5rem 4rem; }
h1 { font-size:1.7rem; margin:0 0 0.3rem; }
h2 { font-size:1.05rem; margin:2.2rem 0 0.8rem; color:var(--accent); }
.sub { color:var(--muted); font-size:0.9rem; margin:0 0 2rem; }
.question { background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:1rem 1.2rem; font-size:1.05rem; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:0.75rem; margin:1.2rem 0; }
.metric { background:var(--panel); border:1px solid var(--border); border-radius:6px;
  padding:0.75rem 0.85rem; }
.metric b { display:block; font-size:1.3rem; color:var(--accent);
  font-variant-numeric:tabular-nums; }
.metric span { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.08em;
  text-transform:uppercase; color:var(--muted); }
.metric small { display:block; color:var(--muted); font-size:0.75rem; margin-top:0.15rem; }
img.chart { width:100%; border-radius:6px; border:1px solid var(--border);
  background:#fff; margin:0.9rem 0; display:block; }
.summary { font-size:1rem; }
.badge { font-family:var(--mono); font-size:0.75rem; color:var(--success); }
.lessons { font-family:var(--mono); font-size:0.78rem; color:var(--muted); }
pre { background:#100d09; border:1px solid var(--border); border-radius:6px;
  padding:0.9rem 1rem; overflow-x:auto; font-family:var(--mono); font-size:0.75rem;
  line-height:1.5; }
details summary { cursor:pointer; color:var(--muted); font-family:var(--mono);
  font-size:0.85rem; }
.attempt { border:1px solid var(--border); border-left:3px solid var(--danger);
  border-radius:6px; padding:0.8rem 1rem; margin:0.7rem 0; font-size:0.85rem; }
.attempt.review { border-left-color:var(--accent); }
.attempt .err { font-family:var(--mono); color:var(--danger); font-size:0.75rem;
  white-space:pre-wrap; }
footer { color:var(--muted); font-family:var(--mono); font-size:0.7rem;
  text-align:center; margin-top:3rem; }
"""


def _chart_tag(path: str) -> str:
    try:
        blob = base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return ""
    return f'<img class="chart" alt="chart" src="data:image/png;base64,{blob}">'


def build_report(job: dict[str, Any]) -> str:
    """Render a finished job as one self-contained HTML document."""
    e = html.escape
    mode_label = "Automated exploration" if job.get("mode") == "eda" else "Question"

    metrics_html = "".join(
        f'<div class="metric"><b>{e(str(m["value"]))}</b><span>{e(str(m["label"]))}</span>'
        + (f'<small>{e(str(m.get("detail", "")))}</small>' if m.get("detail") else "")
        + "</div>"
        for m in job.get("metrics", [])
    )

    # In EDA mode the summary tends to be multi-paragraph; interleave paragraphs
    # with charts so the report reads as a story rather than a dump.
    charts = [c for c in (job.get("charts") or []) if Path(c).exists()]
    paragraphs = [p.strip() for p in job.get("result_summary", "").split("\n\n") if p.strip()]
    story: list[str] = []
    blocks = max(len(charts), len(paragraphs))
    for i in range(blocks):
        if i < len(charts):
            story.append(_chart_tag(charts[i]))
        if i < len(paragraphs):
            story.append(f'<p class="summary">{e(paragraphs[i])}</p>')

    attempts_html = "".join(
        f'<div class="attempt{" review" if h["stderr"].startswith("[quality review]") else ""}">'
        f'<div class="err">{e(h["stderr"][:400])}</div>'
        f"<p>→ {e(h['critique'])}</p></div>"
        for h in job.get("history", [])
    )

    sections = [
        f"<h1>DataMedic report</h1>",
        f'<p class="sub">{e(mode_label)} · generated {date.today().isoformat()}</p>',
        f'<div class="question">{e(job.get("question", ""))}</div>',
    ]
    if metrics_html:
        sections += ["<h2>Key numbers</h2>", f'<div class="metrics">{metrics_html}</div>']
    sections += ["<h2>Findings</h2>", *story]
    if job.get("verified"):
        sections.append('<p class="badge">✓ every figure fact-checked against the analysis output</p>')
    if job.get("lessons_used"):
        used = "; ".join(job["lessons_used"])
        sections.append(f'<p class="lessons">📚 applied learned lessons: {e(used)}</p>')
    if attempts_html:
        sections += ["<h2>How the agent got there</h2>", attempts_html]
    if job.get("plan"):
        sections += ["<h2>Method</h2>", f'<p class="summary">{e(job["plan"])}</p>']
    if job.get("code"):
        sections.append(
            f"<details><summary>Analysis code</summary><pre>{e(job['code'])}</pre></details>"
        )
    sections.append("<footer>made with DataMedic — self-healing data analyst agent</footer>")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>DataMedic report</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(sections)}</main></body></html>"
    )
