"""FastAPI app: routes, background job execution, and static frontend serving."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app.graph import AgentState, build_graph  # noqa: E402
from app.nodes import suggest_questions  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ATTEMPTS = 4

JOBS_DIR = Path(tempfile.gettempdir()) / "datamedic_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store. Known limitation (see README): lost on restart, single process only.
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="DataMedic")
_graph = build_graph()


def _initial_state(csv_path: str, question: str) -> AgentState:
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
        "chart_path": None,
        "charts": [],
        "metrics": [],
        "reviews_used": 0,
        "verified": False,
        "result_summary": "",
        "history": [],
        "status": "planning",
    }


def _run_job(job_id: str, csv_path: str, question: str) -> None:
    try:
        for state in _graph.stream(_initial_state(csv_path, question), stream_mode="values"):
            JOBS[job_id] = state
    except Exception as exc:  # keep the job store consistent even on an unhandled error
        job = JOBS.get(job_id, _initial_state(csv_path, question))
        job["status"] = "failed"
        job["stderr"] = f"{job.get('stderr', '')}\nunhandled error: {exc}".strip()
        JOBS[job_id] = job


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
) -> dict[str, str]:
    contents = await _read_csv_upload(file)

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    csv_path = job_dir / "data.csv"
    csv_path.write_bytes(contents)

    JOBS[job_id] = _initial_state(str(csv_path), question)
    background_tasks.add_task(_run_job, job_id, str(csv_path), question)

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
        "code": job["code"],
        "history": job["history"],
        "chart_urls": [
            f"/chart/{job_id}/{i}" for i in range(len(job.get("charts", [])))
        ],
    }


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
