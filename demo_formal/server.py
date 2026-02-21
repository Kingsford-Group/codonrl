# server.py
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

from multiprocessing import Process

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from worker import run_job  # <-- worker.py in same folder

from pathlib import Path
import json

# ---- hardcoded model assets (server-side only) ----
MODEL_DIR = Path("/mnt/c/CMU/carlLab/codonrl/demo_formal")   # <-- change to your real folder
CKPT_PATH = MODEL_DIR / "ckpt_best_objective.pth"                # <-- your checkpoint filename
SUMMARY_PATH = MODEL_DIR / "training_summary.json"     # <-- your summary/config filename
CSC_PATH = MODEL_DIR / "csc.json"

VERSION = "demo-server/0.2"

OBJECTIVES = [
    {"id": "cai", "description": "Codon Adaptation Index (prefix-aware)"},
    {"id": "csc", "description": "Codon Stability Coefficient (prefix-aware)"},
    {"id": "gc",  "description": "GC content (match target or maximize)"},
    {"id": "u",   "description": "Uridine/T minimization (match target or minimize)"},
]

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

CFG = load_json(SUMMARY_PATH)['config']         # cfg dict, server-owned
CSC_WEIGHTS = load_json(CSC_PATH)      # dict codon->weight (or whatever your worker expects)

@dataclass
class Job:
    id: str
    status: str  # running/success/error/aborted
    start_time: float
    end_time: Optional[float] = None
    protein: Optional[str] = None
    objectives: Optional[Dict[str, Any]] = None
    result_mrna: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None

JOBS: Dict[str, Job] = {}

class OptimizationRequest(BaseModel):
    sequence: str                 # protein sequence
    objectives: Dict[str, Any]    # includes decode params + (cfg, ckpt, csc_weights) depending on your choice

class OptimizationCreateResponse(BaseModel):
    id: str

# ---------- Simple UI ----------
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CodonRL Demo Server</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 0;
      min-height: 100vh;

      /* center the whole app container horizontally */
      display: flex;
      justify-content: center;
    }

    /* a centered column container */
    .container {
      width: min(900px, 92vw);
      padding: 24px;

      /* center all text/content by default */
      text-align: center;
    }

    /* center "row" elements (file input + button) */
    .row {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: center; /* <-- key */
      flex-wrap: wrap;
      margin-top: 8px;
    }

    .card {
      border: 1px solid #ccc;
      border-radius: 12px;
      padding: 12px;
      margin: 12px 0;
      min-width: 0;

      /* center card contents */
      text-align: center;
    }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;        /* preserves newlines but allows wrapping */
      overflow-wrap: anywhere;      /* break very long tokens */
      word-break: break-word;       /* fallback */
      max-width: 100%;
      text-align: left;
    }
    .seqbox {
      margin-top: 14px;
      text-align: left;

      white-space: pre-wrap;
      word-break: break-word;     /* same as the working page */
      overflow-wrap: anywhere;    /* extra safety */

      background: #fafafa;
      border: 1px solid #eee;
      border-radius: 12px;
      padding: 14px;

      max-width: 100%;
      overflow: auto;             /* keeps it inside the card */
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
      min-width: 0;
    }
    .outbox {
      width: 100%;
      box-sizing: border-box;
      word-break: break-all;      /* force break inside long RNA strings */
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      overflow-x: hidden;         /* or 'auto' if you prefer a scrollbar */
    }

    .small { color: #555; font-size: 12px; }

    button {
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid #aaa;
      background: #f7f7f7;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="container">
  <h2>CodonRL Optimization Demo</h2>

  <div class="row">
    <input type="file" id="fasta" accept=".fa,.fasta,.txt"/>
    <button onclick="submitJob()">Submit optimization</button>
  </div>

  <div class="small">
    Multi-job demo: every submission creates its own status block and polls independently.
  </div>

  <div id="jobs"></div>
  


<script>
function parseFasta(text) {
  const lines = text.split(/\r?\n/).filter(x => x.trim().length > 0);
  return lines.filter(l => !l.startsWith('>')).join('').trim();
}

async function submitJob() {
  const file = document.getElementById("fasta").files[0];
  if (!file) { alert("Please choose a FASTA file."); return; }
  const text = await file.text();
  const protein = parseFasta(text);
  if (!protein) { alert("No sequence found in FASTA."); return; }

  // Demo objective config: adjust freely
  // IMPORTANT: for a runnable end-to-end setup you must supply cfg/ckpt/csc_weights
  const body = {
    sequence: protein,
    objectives: {
      // decode params
      alpha_cai: 1.0,
      alpha_csc: 1.0,
      alpha_gc:  1.0,
      alpha_u:   0.5,
      target_gc: 0.55,

      // TODO: provide these in whichever way you choose:
      // cfg: {...},
      // ckpt: "/abs/path/to/checkpoint.pt",
      // csc_weights: {...} or a file path you load on server side
    }
  };

  const res = await fetch("/optimizations", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    alert("Submission failed: " + await res.text());
    return;
  }

  const data = await res.json();
  createJobCard(data.id, protein.length);
}

function createJobCard(id, L) {
  const container = document.getElementById("jobs");
  const div = document.createElement("div");
  div.className = "card";
  div.id = "job-" + id;
  div.innerHTML = `
    <div class="row">
      <b>Job</b> <span class="mono">${id}</span>
      <span class="small">Protein length: ${L}</span>
    </div>
    <div class="mono" id="job-status-${id}">Job submitted and running, please wait...</div>
    <div class="mono outbox" id="job-result-${id}"></div>
  `;
  container.prepend(div);
  pollJob(id);
}

async function pollJob(id) {
  const statusEl = document.getElementById("job-status-" + id);
  const resultEl = document.getElementById("job-result-" + id);

  const timer = setInterval(async () => {
    const res = await fetch(`/optimizations/${id}/status`);
    if (!res.ok) return;
    const st = await res.json();

    statusEl.textContent = `Status: ${st.status}` + (st.error ? `\nError: ${st.error}` : "");

    if (st.status === "success") {
      clearInterval(timer);
      const rr = await fetch(`/optimizations/${id}/result`);
      const out = await rr.json();
      resultEl.textContent = "\nOptimized mRNA:\n" + out.sequence;
    } else if (st.status === "error" || st.status === "aborted") {
      clearInterval(timer);
    }
  }, 1500);
}
</script>
  </div>
</body>
</html>
"""

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML

@app.get("/health")
def health():
    running = sum(1 for j in JOBS.values() if j.status == "running")
    status = "available"  # for demo we are always available; real service may return busy
    return {"status": status, "version": VERSION, "running_jobs": running}

@app.get("/objectives")
def objectives():
    return OBJECTIVES

@app.post("/optimizations", response_model=OptimizationCreateResponse)
def create_optimization(req: OptimizationRequest):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = Job(
        id=job_id,
        status="running",
        start_time=time.time(),
        protein=req.sequence,
        objectives=req.objectives,
    )

    # IMPORTANT: worker needs to call back to this server.
    # If you run on a different host/port, update this.
    server_base_url = "http://127.0.0.1:8000"

    # In this demo we pass everything through objectives.
    # You can switch to server-side config later.
    cfg = CFG
    ckpt_path = str(CKPT_PATH)
    csc_weights = CSC_WEIGHTS
    decode_params = req.objectives  # only the alphas/targets from the user


    if cfg is None or ckpt_path is None or csc_weights is None:
        raise HTTPException(
            400,
            "Missing required fields in objectives: cfg, ckpt, csc_weights "
            "(for demo). You can also hardcode these server-side."
        )

    p = Process(
        target=run_job,
        kwargs=dict(
            server_base_url=server_base_url,
            job_id=job_id,
            protein=req.sequence,
            cfg=cfg,
            ckpt_path=ckpt_path,
            csc_weights=csc_weights,
            decode_params=decode_params,
        ),
    )
    p.daemon = True
    p.start()
    JOBS[job_id].pid = p.pid

    return {"id": job_id}

@app.get("/optimizations/{job_id}/status")
def optimization_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "start_time": job.start_time,
        "end_time": job.end_time,
        "error": job.error,
        "pid": job.pid,
    }

@app.get("/optimizations/{job_id}/result")
def optimization_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "success":
        raise HTTPException(409, f"Job not finished (status={job.status})")
    return {"id": job.id, "sequence": job.result_mrna}

@app.put("/optimizations/{job_id}/abort")
def abort_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "running":
        return {"ok": True, "status": job.status}
    job.status = "aborted"
    job.end_time = time.time()
    return {"ok": True, "status": "aborted"}

# ---- Internal callbacks from worker processes ----
@app.post("/internal/optimizations/{job_id}/complete")
def internal_complete(job_id: str, payload: dict):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job.result_mrna = payload.get("sequence", "")
    job.status = "success"
    job.end_time = time.time()
    return {"ok": True}

@app.post("/internal/optimizations/{job_id}/fail")
def internal_fail(job_id: str, payload: dict):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job.status = "error"
    job.error = payload.get("error", "unknown error")
    job.end_time = time.time()
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
