(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileNameEl = document.getElementById("file-name");
  const dropzoneText = document.getElementById("dropzone-text");
  const questionEl = document.getElementById("question");
  const form = document.getElementById("analyze-form");
  const submitBtn = document.getElementById("submit-btn");
  const raceCheckbox = document.getElementById("race-checkbox");

  const suggestionsEl = document.getElementById("suggestions");

  const cockpitPanel = document.getElementById("cockpit-panel");
  const raceLanesEl = document.getElementById("race-lanes");
  const streamNodeEl = document.getElementById("stream-node");
  const streamTextEl = document.getElementById("stream-text");
  const lessonsChip = document.getElementById("lessons-chip");

  const statusPanel = document.getElementById("status-panel");
  const statusPill = document.getElementById("status-pill");
  const attemptMeter = document.getElementById("attempt-meter");
  const attemptLabel = document.getElementById("attempt-label");
  const attemptsEl = document.getElementById("attempts");

  const resultPanel = document.getElementById("result-panel");
  const metricsEl = document.getElementById("metrics");
  const chartsEl = document.getElementById("charts");
  const summaryEl = document.getElementById("summary");
  const verifiedBadge = document.getElementById("verified-badge");
  const memoryNote = document.getElementById("memory-note");
  const reportLink = document.getElementById("report-link");
  const finalCodeEl = document.getElementById("final-code");
  const copyBtn = document.getElementById("copy-btn");

  let selectedFile = null;
  let renderedAttempts = 0;
  let pollHandle = null;
  let eventSource = null;

  // ---------- drag & drop ----------

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      fileNameEl.textContent = "Only .csv files are supported";
      fileNameEl.style.color = "var(--danger)";
      selectedFile = null;
      return;
    }
    selectedFile = file;
    fileNameEl.style.color = "var(--success)";
    fileNameEl.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    dropzoneText.textContent = "Drop a different CSV, or click to browse";
    fetchSuggestions(file);
  }

  // ---------- suggested questions ----------

  async function fetchSuggestions(file) {
    suggestionsEl.innerHTML = `<span class="suggestions-hint">Reading your data for question ideas…</span>`;
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/suggest", { method: "POST", body });
      const { questions } = await res.json();
      if (file !== selectedFile) return; // a newer file was chosen meanwhile
      suggestionsEl.innerHTML = "";
      for (const q of questions) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = q;
        chip.addEventListener("click", () => {
          questionEl.value = q;
          questionEl.focus();
        });
        suggestionsEl.appendChild(chip);
      }
    } catch {
      suggestionsEl.innerHTML = "";
    }
  }

  // ---------- submit ----------

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      fileNameEl.style.color = "var(--danger)";
      fileNameEl.textContent = "Choose a CSV first";
      return;
    }
    const question = questionEl.value.trim(); // empty = auto-EDA mode

    setBusy(true);
    resetPanels();
    pulse.setMode("running");

    const body = new FormData();
    body.append("file", selectedFile);
    body.append("question", question);
    body.append("race", raceCheckbox.checked ? "2" : "1");

    try {
      const res = await fetch("/analyze", { method: "POST", body });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "request failed");
      }
      const { job_id } = await res.json();
      cockpitPanel.classList.remove("hidden");
      statusPanel.classList.remove("hidden");
      openEvents(job_id);
      startPolling(job_id);
    } catch (err) {
      setBusy(false);
      pulse.setMode("failed");
      fileNameEl.style.color = "var(--danger)";
      fileNameEl.textContent = `Error: ${err.message}`;
    }
  });

  function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.classList.toggle("busy", busy);
    submitBtn.querySelector(".btn-label").textContent = busy ? "Analyzing" : "Run analysis";
  }

  function resetPanels() {
    renderedAttempts = 0;
    attemptsEl.innerHTML = "";
    resultPanel.classList.add("hidden");
    raceLanesEl.classList.add("hidden");
    raceLanesEl.innerHTML = "";
    lessonsChip.classList.add("hidden");
    streamTextEl.textContent = "";
    streamNodeEl.textContent = "waiting…";
    document.querySelectorAll(".gnode").forEach((n) => n.classList.remove("active", "visited"));
    document.querySelectorAll(".edge").forEach((n) => n.classList.remove("lit"));
    if (eventSource) eventSource.close();
  }

  // ---------- live cockpit (SSE) ----------

  const NODE_TO_SVG = {
    profile_csv: "n-profile_csv",
    plan: "n-plan",
    write_code: "n-code",
    race: "n-code",
    execute: "n-execute",
    review: "n-review",
    critique: "n-critique",
    summarize: "n-summarize",
    verify: "n-verify",
    learn: "n-learn",
  };

  const NODE_LABELS = {
    plan: "planner · writing the analysis plan",
    write_code: "coder · writing python",
    race: "race · rival coders running",
    critique: "debugger · diagnosing the failure",
    summarize: "writer · drafting the insight",
    profile_csv: "profiling the CSV",
    execute: "sandbox · running the script",
    review: "reviewer · judging quality",
    verify: "fact-checker · verifying numbers",
    learn: "librarian · distilling lessons",
  };

  const STREAMING_NODES = new Set(["plan", "write_code", "critique", "summarize"]);
  const SILENT_PLACEHOLDERS = {
    execute: "…running the script (no LLM call — deterministic)",
    race: "…rival coders writing in parallel, see lanes above",
  };
  const MAX_STREAM_CHARS = 8000;

  function setActiveNode(node) {
    document.querySelectorAll(".gnode.active").forEach((n) => n.classList.remove("active"));
    const svgId = NODE_TO_SVG[node];
    if (svgId) document.getElementById(svgId)?.classList.add("active");
  }

  function openEvents(jobId) {
    eventSource = new EventSource(`/events/${jobId}`);
    eventSource.onmessage = (msg) => {
      let ev;
      try {
        ev = JSON.parse(msg.data);
      } catch {
        return;
      }
      handleEvent(ev);
    };
    eventSource.onerror = () => {
      /* polling still drives the vitals panel; cockpit just freezes */
    };
  }

  function handleEvent(ev) {
    switch (ev.kind) {
      case "node_start": {
        setActiveNode(ev.node);
        streamNodeEl.textContent = NODE_LABELS[ev.node] || ev.node;
        // Always clear — otherwise a non-streaming node (review/verify/learn)
        // would keep showing leftover text from the last streaming node.
        streamTextEl.textContent = STREAMING_NODES.has(ev.node)
          ? ""
          : SILENT_PLACEHOLDERS[ev.node] || "…thinking";
        break;
      }
      case "node_end": {
        const svgId = NODE_TO_SVG[ev.node];
        if (svgId) {
          const el = document.getElementById(svgId);
          el?.classList.remove("active");
          el?.classList.add("visited");
        }
        break;
      }
      case "token": {
        streamTextEl.textContent = (streamTextEl.textContent + ev.text).slice(-MAX_STREAM_CHARS);
        streamTextEl.scrollTop = streamTextEl.scrollHeight;
        break;
      }
      case "race_start": {
        raceLanesEl.classList.remove("hidden");
        raceLanesEl.innerHTML = "";
        for (let i = 0; i < ev.n; i++) {
          const lane = document.createElement("div");
          lane.className = "race-lane";
          lane.id = `lane-${i}`;
          lane.innerHTML = `
            <span class="lane-name">coder ${String.fromCharCode(65 + i)}</span>
            <span class="lane-strategy">${escapeHtml(shortStrategy(ev.strategies[i]))}</span>
            <span class="lane-phase" id="lane-phase-${i}">queued</span>`;
          raceLanesEl.appendChild(lane);
        }
        break;
      }
      case "race_candidate": {
        const phase = document.getElementById(`lane-phase-${ev.index}`);
        if (phase) {
          phase.textContent = { writing: "writing…", running: "running…", passed: "✓ passed", crashed: "✗ crashed" }[ev.phase] || ev.phase;
          phase.className = `lane-phase phase-${ev.phase}`;
        }
        break;
      }
      case "race_end": {
        if (ev.winner !== null && ev.winner !== undefined) {
          document.getElementById(`lane-${ev.winner}`)?.classList.add("winner");
          const note = document.createElement("p");
          note.className = "race-reason";
          note.textContent = ev.reason ? `judge: ${ev.reason}` : "winner chosen";
          raceLanesEl.appendChild(note);
        }
        break;
      }
      case "lessons_applied": {
        lessonsChip.classList.remove("hidden");
        lessonsChip.textContent = `📚 ${ev.lessons.length} learned lesson${ev.lessons.length > 1 ? "s" : ""} applied`;
        lessonsChip.title = ev.lessons.join("\n");
        break;
      }
      case "lessons_learned": {
        lessonsChip.classList.remove("hidden");
        lessonsChip.textContent = `🧠 learned ${ev.count} new lesson${ev.count > 1 ? "s" : ""}`;
        break;
      }
      case "job_done": {
        eventSource?.close();
        break;
      }
    }
  }

  function shortStrategy(s) {
    if (!s) return "";
    if (s.toLowerCase().includes("vectorized")) return "vectorized";
    if (s.toLowerCase().includes("defensive")) return "defensive";
    if (s.toLowerCase().includes("statistics")) return "stats-first";
    return s.slice(0, 24);
  }

  // ---------- polling (vitals + result) ----------

  function startPolling(jobId) {
    if (pollHandle) clearInterval(pollHandle);
    poll(jobId);
    pollHandle = setInterval(() => poll(jobId), 1200);
  }

  async function poll(jobId) {
    let data;
    try {
      const res = await fetch(`/status/${jobId}`);
      if (!res.ok) return;
      data = await res.json();
    } catch {
      return;
    }

    renderStatus(data);
    renderAttempts(data.history);

    if (data.status === "done") {
      clearInterval(pollHandle);
      setBusy(false);
      pulse.setMode("done");
      const res = await fetch(`/result/${jobId}`);
      const result = await res.json();
      renderResult(result);
    } else if (data.status === "failed") {
      clearInterval(pollHandle);
      setBusy(false);
      pulse.setMode("failed");
      if (data.error && !data.history.length) {
        const card = document.createElement("div");
        card.className = "attempt-card";
        card.innerHTML = `<h3>Run failed before any attempt</h3>
          <div class="stderr-snippet">${escapeHtml(truncate(data.error, 500))}</div>`;
        attemptsEl.appendChild(card);
      }
    }
  }

  const STATUS_LABELS = {
    planning: "Planning",
    coding: "Writing code",
    executing: "Executing",
    reviewing: "Reviewing quality",
    summarizing: "Writing summary",
    verifying: "Fact-checking",
    fixing: "Fixing",
    done: "Done",
    failed: "Failed",
  };

  function renderStatus(data) {
    const label = STATUS_LABELS[data.status] || data.status;
    statusPill.textContent = label;
    statusPill.className = "pill " + (data.status === "done" ? "done" : data.status === "failed" ? "failed" : "active");

    attemptLabel.textContent = `attempt ${Math.min(data.attempt, data.max_attempts)} / ${data.max_attempts}`;

    attemptMeter.innerHTML = "";
    for (let i = 0; i < data.max_attempts; i++) {
      const dot = document.createElement("span");
      if (i < data.history.length) {
        dot.classList.add("filled-bad");
      } else if (i === data.history.length && data.status === "done") {
        dot.classList.add("filled-ok");
      }
      attemptMeter.appendChild(dot);
    }
  }

  function renderAttempts(history) {
    for (let i = renderedAttempts; i < history.length; i++) {
      const h = history[i];
      const isReview = (h.stderr || "").startsWith("[quality review]");
      const card = document.createElement("div");
      card.className = "attempt-card" + (isReview ? " review-card" : "");
      card.innerHTML = `
        <h3>${isReview ? `Attempt ${i + 1} sent back by reviewer` : `Attempt ${i + 1} failed`}</h3>
        <details>
          <summary>Show code</summary>
          <pre><code>${escapeHtml(h.code)}</code></pre>
        </details>
        <div class="stderr-snippet">${escapeHtml(truncate(h.stderr, 400))}</div>
        <p class="critique-line">${escapeHtml(h.critique)}</p>
      `;
      attemptsEl.appendChild(card);
    }
    renderedAttempts = history.length;
  }

  function renderResult(result) {
    resultPanel.classList.remove("hidden");

    metricsEl.innerHTML = "";
    for (const m of result.metrics || []) {
      const card = document.createElement("div");
      card.className = "metric-card";
      card.innerHTML = `
        <span class="metric-value">${escapeHtml(m.value)}</span>
        <span class="metric-label">${escapeHtml(m.label)}</span>
        ${m.detail ? `<span class="metric-detail">${escapeHtml(m.detail)}</span>` : ""}
      `;
      metricsEl.appendChild(card);
    }

    chartsEl.innerHTML = "";
    const urls = result.chart_urls || [];
    chartsEl.classList.toggle("two-col", urls.length > 1);
    for (const url of urls) {
      const img = document.createElement("img");
      img.className = "chart";
      img.alt = "Generated chart";
      img.src = url + `?t=${Date.now()}`;
      chartsEl.appendChild(img);
    }

    summaryEl.textContent = result.result_summary;
    verifiedBadge.classList.toggle("hidden", !result.verified);

    const notes = [];
    if ((result.lessons_used || []).length) {
      notes.push(`📚 applied ${result.lessons_used.length} learned lesson${result.lessons_used.length > 1 ? "s" : ""}: ${result.lessons_used.join("; ")}`);
    }
    if (result.lessons_learned > 0) {
      notes.push(`🧠 learned ${result.lessons_learned} new lesson${result.lessons_learned > 1 ? "s" : ""} for future runs`);
    }
    memoryNote.textContent = notes.join(" · ");
    memoryNote.classList.toggle("hidden", notes.length === 0);

    reportLink.href = result.report_url;
    finalCodeEl.textContent = result.code;
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s ?? "";
    return div.innerHTML;
  }

  // ---------- copy button ----------

  copyBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      await navigator.clipboard.writeText(finalCodeEl.textContent);
      copyBtn.textContent = "Copied";
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("copied");
      }, 1400);
    } catch {
      /* clipboard unavailable — no-op */
    }
  });

  // ---------- ambient pulse-line ----------

  const pulse = (() => {
    const canvas = document.getElementById("pulse-line");
    const ctx = canvas.getContext("2d");
    let mode = "idle"; // idle | running | done | failed
    let t = 0;
    let width = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const colors = {
      idle: "#a9762c",
      running: "#e8a23d",
      done: "#4cbb82",
      failed: "#e8604a",
    };

    function resize() {
      width = canvas.clientWidth;
      canvas.width = width * dpr;
      canvas.height = canvas.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function waveY(x, phase) {
      const w = canvas.clientHeight;
      const mid = w / 2;
      const speed = mode === "running" ? 3.2 : mode === "failed" ? 4 : 1.2;
      const period = mode === "failed" ? 90 : 160;
      const local = ((x + phase * speed) % period) / period;

      let spike = 0;
      if (local > 0.42 && local < 0.58) {
        const p = (local - 0.42) / 0.16;
        if (mode === "failed") {
          spike = Math.sin(p * Math.PI * 6) * (1 - Math.abs(p - 0.5) * 2) * 0.9;
        } else {
          spike = Math.sin(p * Math.PI) * (mode === "idle" ? 0.35 : 0.85);
        }
      }
      const jitter = mode === "running" ? Math.sin(x * 0.4 + phase * 2) * 0.03 : 0;
      return mid - spike * (mid * 0.85) - jitter * mid;
    }

    function draw() {
      ctx.clearRect(0, 0, width, canvas.clientHeight);
      ctx.beginPath();
      ctx.strokeStyle = colors[mode];
      ctx.lineWidth = 1.6;
      ctx.shadowColor = colors[mode];
      ctx.shadowBlur = 6;
      for (let x = 0; x <= width; x += 2) {
        const y = waveY(x, t);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      t += 1;
      requestAnimationFrame(draw);
    }

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduceMotion) requestAnimationFrame(draw);
    else {
      ctx.beginPath();
      ctx.strokeStyle = colors.idle;
      ctx.moveTo(0, canvas.clientHeight / 2);
      ctx.lineTo(width, canvas.clientHeight / 2);
      ctx.stroke();
    }

    return {
      setMode(m) {
        mode = m;
      },
    };
  })();
})();
