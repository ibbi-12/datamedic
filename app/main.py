"""FastAPI app: routes, background job execution, SSE, and static frontend serving."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app import events  # noqa: E402
from app.graph import AgentState, build_graph  # noqa: E402
from app.nodes import CSVLoadError, suggest_questions  # noqa: E402
from app.report import build_report  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ATTEMPTS = 4

JOBS_DIR = Path(tempfile.gettempdir()) / "datamedic_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store. Known limitation (see README): lost on restart, single process only.
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="DataMedic")
_graph = build_graph()


def _initial_state(csv_path: str, question: str, race_n: int = 1) -> AgentState:
    return {
        "csv_path": csv_path,
        "csv_profile": "",
        "question": question,
        "mode": "question",
        "plan": "",
        "code": "",
        "stdout": "",
        "stderr": "",
        "attempt": 0,
        "max_attempts": MAX_ATTEMPTS,
        "race_n": race_n,
        "race_report": {},
        "chart_path": None,
        "charts": [],
        "metrics": [],
        "lessons_used": [],
        "lessons_learned": 0,
        "reviews_used": 0,
        "verified": False,
        "result_summary": "",
        "history": [],
        "status": "planning",
    }


def _run_job(job_id: str, csv_path: str, question: str, race_n: int) -> None:
    stream = events.STREAMS.get(job_id)
    events.bind(stream)
    try:
        initial = _initial_state(csv_path, question, race_n)
        for state in _graph.stream(initial, stream_mode="values"):
            JOBS[job_id] = state
    except Exception as exc:  # keep the job store consistent even on an unhandled error
        logging.exception("job %s failed with unhandled error", job_id)
        job = JOBS.get(job_id, _initial_state(csv_path, question, race_n))
        job["status"] = "failed"
        prefix = "could not read this CSV" if isinstance(exc, CSVLoadError) else "unhandled error"
        job["stderr"] = f"{job.get('stderr', '')}\n{prefix}: {exc}".strip()
        JOBS[job_id] = job
    finally:
        events.bind(None)
        if stream is not None:
            stream.emit("job_done", status=JOBS.get(job_id, {}).get("status", "failed"))
            stream.close()


async def _read_csv_upload(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a .csv")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="file exceeds 10 MB limit")
    if not contents:
        raise HTTPException(status_code=400, detail="file is empty")
    return contents


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = Form(""),
    race: int = Form(1),
) -> dict[str, str]:
    contents = await _read_csv_upload(file)
    race_n = max(1, min(race, 3))

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    csv_path = job_dir / "data.csv"
    csv_path.write_bytes(contents)

    JOBS[job_id] = _initial_state(str(csv_path), question, race_n)
    events.create_stream(job_id)
    background_tasks.add_task(_run_job, job_id, str(csv_path), question, race_n)

    return {"job_id": job_id}


@app.post("/suggest")
async def suggest(file: UploadFile = File(...)) -> dict[str, list[str]]:
    """Propose 3 questions the uploaded CSV can answer (one LLM call, no job created)."""
    contents = await _read_csv_upload(file)
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            questions = await run_in_threadpool(suggest_questions, tmp.name)
        except Exception:
            questions = []  # suggestions are a nicety; never block the upload flow
    return {"questions": questions}


@app.get("/events/{job_id}")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events: replay this job's event log, then follow it live."""
    if job_id not in events.STREAMS and job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        sent = 0
        while True:
            stream = events.STREAMS.get(job_id)
            if stream is None:
                break
            pending = stream.events[sent:]
            for event in pending:
                yield f"data: {json.dumps(event)}\n\n"
            sent += len(pending)
            if stream.closed and sent >= len(stream.events):
                break
            await asyncio.sleep(0.12)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "status": job["status"],
        "attempt": job["attempt"],
        "max_attempts": job["max_attempts"],
        "history": job["history"],
        "error": job.get("stderr", "") if job["status"] == "failed" else "",
    }


@app.get("/result/{job_id}")
async def result(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "status": job["status"],
        "mode": job.get("mode", "question"),
        "question": job["question"],
        "result_summary": job["result_summary"],
        "metrics": job.get("metrics", []),
        "verified": job.get("verified", False),
        "lessons_used": job.get("lessons_used", []),
        "lessons_learned": job.get("lessons_learned", 0),
        "race_report": job.get("race_report", {}),
        "code": job["code"],
        "history": job["history"],
        "chart_urls": [
            f"/chart/{job_id}/{i}" for i in range(len(job.get("charts", [])))
        ],
        "report_url": f"/report/{job_id}",
    }


@app.get("/report/{job_id}")
async def report(job_id: str) -> HTMLResponse:
    """A finished job as one self-contained HTML file (charts inlined)."""
    job = JOBS.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="finished job not found")
    html_doc = build_report(job)
    return HTMLResponse(
        html_doc,
        headers={"Content-Disposition": f'inline; filename="datamedic-report-{job_id[:8]}.html"'},
    )


@app.get("/chart/{job_id}/{index}")
async def chart(job_id: str, index: int) -> FileResponse:
    job = JOBS.get(job_id)
    charts = (job or {}).get("charts", [])
    if job is None or index < 0 or index >= len(charts):
        raise HTTPException(status_code=404, detail="chart not found")
    chart_path = Path(charts[index])
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="chart not found")
    return FileResponse(chart_path, media_type="image/png")


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
