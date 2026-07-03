(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileNameEl = document.getElementById("file-name");
  const dropzoneText = document.getElementById("dropzone-text");
  const questionEl = document.getElementById("question");
  const form = document.getElementById("analyze-form");
  const submitBtn = document.getElementById("submit-btn");

  const statusPanel = document.getElementById("status-panel");
  const statusPill = document.getElementById("status-pill");
  const attemptMeter = document.getElementById("attempt-meter");
  const attemptLabel = document.getElementById("attempt-label");
  const attemptsEl = document.getElementById("attempts");

  const resultPanel = document.getElementById("result-panel");
  const chartImg = document.getElementById("chart-img");
  const summaryEl = document.getElementById("summary");
  const finalCodeEl = document.getElementById("final-code");
  const copyBtn = document.getElementById("copy-btn");

  let selectedFile = null;
  let renderedAttempts = 0;
  let pollHandle = null;

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
  }

  // ---------- submit ----------

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      fileNameEl.style.color = "var(--danger)";
      fileNameEl.textContent = "Choose a CSV first";
      return;
    }
    const question = questionEl.value.trim();
    if (!question) {
      questionEl.focus();
      return;
    }

    setBusy(true);
    resetPanels();
    pulse.setMode("running");

    const body = new FormData();
    body.append("file", selectedFile);
    body.append("question", question);

    try {
      const res = await fetch("/analyze", { method: "POST", body });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "request failed");
      }
      const { job_id } = await res.json();
      statusPanel.classList.remove("hidden");
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
  }

  // ---------- polling ----------

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
    }
  }

  const STATUS_LABELS = {
    planning: "Planning",
    coding: "Writing code",
    executing: "Executing",
    summarizing: "Writing summary",
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
      const card = document.createElement("div");
      card.className = "attempt-card";
      card.innerHTML = `
        <h3>Attempt ${i + 1} failed</h3>
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
    if (result.chart_url) {
      chartImg.src = result.chart_url + `?t=${Date.now()}`;
      chartImg.style.display = "block";
    } else {
      chartImg.style.display = "none";
    }
    summaryEl.textContent = result.result_summary;
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
