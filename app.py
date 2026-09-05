from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import mpmath as mp
from flint import acb, ctx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ollama import Client

from checkpoint import load_checkpoint, save_checkpoint


START_N = 10_000_000_000_001
DB = os.getenv("RH_DB_PATH", "riemann_lab.sqlite3")
VERIFY_DPS = int(os.getenv("RH_VERIFY_DPS", "160"))
FLINT_SEARCH_DPS = int(os.getenv("RH_FLINT_SEARCH_DPS", "50"))
BATCH = max(1, int(os.getenv("RH_BATCH_SIZE", "8")))
FLINT_THREADS = max(1, int(os.getenv("RH_FLINT_THREADS", "1")))
MAX_N = int(os.getenv("RH_MAX_N", "0"))
OFFLINE_EVERY_BATCHES = max(1, int(os.getenv("RH_OFFLINE_SCAN_BATCHES", "25")))
CHECKPOINT_EVERY_BATCHES = max(1, int(os.getenv("RH_CHECKPOINT_BATCHES", "4")))
INTERVAL = max(0.0, float(os.getenv("RH_SEARCH_INTERVAL", "0.02")))
MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY", "")

DISCOVERY = mp.mpf("1e-25")
FINAL = mp.mpf("1e-40")
LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with LOCK:
        c = db()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS zeros(
                id INTEGER PRIMARY KEY,n INTEGER,real_part TEXT,imaginary_part TEXT,
                residual TEXT,deviation TEXT,search_dps INTEGER,verify_dps INTEGER,
                verified INTEGER,suspicious INTEGER,method TEXT,created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS candidates(
                id INTEGER PRIMARY KEY,source TEXT,real_part TEXT,imaginary_part TEXT,
                residual TEXT,deviation TEXT,verify_dps INTEGER,status TEXT,evidence TEXT,created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS reports(
                id INTEGER PRIMARY KEY,candidate_id INTEGER,model TEXT,report TEXT,created_at TEXT
            );
            """
        )
        c.commit()
        c.close()


def state(k: str, default=None):
    if default is None:
        default = str(START_N)
    with LOCK:
        c = db()
        r = c.execute("SELECT value FROM state WHERE key=?", (k,)).fetchone()
        c.close()
        return r["value"] if r else default


def setstate(k: str, v) -> None:
    with LOCK:
        c = db()
        c.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, str(v)),
        )
        c.commit()
        c.close()


def count(table: str) -> int:
    with LOCK:
        c = db()
        n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        c.close()
        return n


def latest(table: str):
    with LOCK:
        c = db()
        r = c.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
        c.close()
        return dict(r) if r else None


def zeros(limit: int = 30):
    with LOCK:
        c = db()
        r = c.execute("SELECT * FROM zeros ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        c.close()
        return [dict(x) for x in r]


def addzero(n, real, imag, residual, deviation, verified, suspicious, method):
    with LOCK:
        c = db()
        c.execute(
            "INSERT INTO zeros VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)",
            (n, real, imag, residual, deviation, FLINT_SEARCH_DPS, VERIFY_DPS,
             int(verified), int(suspicious), method, now()),
        )
        c.commit()
        c.close()


def addcandidate(source, real, imag, residual, deviation, evidence):
    with LOCK:
        c = db()
        cur = c.execute(
            "INSERT INTO candidates VALUES(NULL,?,?,?,?,?,?,?,?,?)",
            (source, real, imag, residual, deviation, VERIFY_DPS,
             "numerically_survived", json.dumps(evidence), now()),
        )
        c.commit()
        candidate_id = cur.lastrowid
        c.close()
        return candidate_id


def addreport(candidate_id, report):
    with LOCK:
        c = db()
        c.execute("INSERT INTO reports VALUES(NULL,?,?,?,?)", (candidate_id, MODEL, report, now()))
        c.commit()
        c.close()


# FLINT/Arb + Platt: optimized native numerical engine.
def configure_flint(dps: int) -> None:
    ctx.dps = dps
    ctx.threads = FLINT_THREADS


def flint_zero_batch(start: int, count_: int):
    configure_flint(FLINT_SEARCH_DPS)
    return list(acb.zeta_zeros(start, count_))


def flint_verify_zero(n: int):
    configure_flint(VERIFY_DPS)
    z = acb.zeta_zero(n)
    residual = z.zeta().abs_upper()
    real = z.real.str(VERIFY_DPS, radius=False, more=True)
    imag = z.imag.str(VERIFY_DPS, radius=False, more=True)
    res = residual.str(70, radius=False, more=True)
    return real, imag, res


# Exploratory only. This searches away from Re(s)=1/2, but is not exhaustive proof.
def exploratory(t):
    with mp.workdps(60):
        for r in (".10", ".20", ".30", ".40", ".45", ".55", ".60", ".70", ".80", ".90"):
            for d in ("-.35", "-.15", "0", ".15", ".35"):
                seed = mp.mpc(r, mp.mpf(t) + mp.mpf(d))
                try:
                    root = mp.findroot(
                        mp.zeta,
                        (seed, seed + mp.mpc(".003", ".003")),
                        solver="secant", tol=mp.mpf("1e-45"), maxsteps=40, verify=False,
                    )
                    residual = abs(mp.zeta(root))
                    deviation = abs(mp.re(root) - mp.mpf(".5"))
                    if (mp.im(root) > 10 and 0 < mp.re(root) < 1
                            and residual < mp.mpf("1e-40") and deviation > DISCOVERY):
                        return root
                except Exception:
                    pass
    return None


def ai_report(evidence):
    if not API_KEY:
        return "OLLAMA_API_KEY is not set. Numerical evidence follows:\n\n" + json.dumps(evidence, indent=2)
    client = Client(host="https://ollama.com", headers={"Authorization": f"Bearer {API_KEY}"})
    system = (
        "You are a mathematical research-report assistant. Explain why the program stopped, preserve "
        "numerical evidence, distinguish numerical evidence from formal proof, and never claim the "
        "Riemann Hypothesis is disproved unless the supplied evidence establishes that. Do not invent facts."
    )
    result = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Generate a cautious research report from this evidence:\n" + json.dumps(evidence, indent=2)},
        ],
    )
    return result["message"]["content"]


async def checkpoint(paused=False, extra=None):
    payload = {
        "next_n": int(state("next_n")), "zeros_checked": count("zeros"),
        "updated_at_utc": now(), "paused": paused, "search_start_n": START_N,
    }
    if extra:
        payload.update(extra)
    try:
        ok = await save_checkpoint(payload)
        if not ok:
            search.error = "GitHub checkpoint is not configured or could not be saved."
        return ok
    except Exception as ex:
        search.error = f"Checkpoint save failed: {type(ex).__name__}: {ex}"
        return False


class Search:
    def __init__(self):
        self.running = False
        self.stopped = False
        self.reason = ""
        self.error = ""
        self.last = None
        self.task = None
        self.batches_done = 0
        self.batch_seconds = None

    def status(self):
        return {
            "running": self.running, "stopped": self.stopped, "reason": self.reason,
            "error": self.error, "start_n": START_N,
            "next_n": int(state("next_n", str(START_N))),
            "current_n": int(state("current_n", state("next_n", str(START_N)))),
            "phase": state("phase", "idle"), "zeros_checked": count("zeros"),
            "candidate_count": count("candidates"), "search_dps": FLINT_SEARCH_DPS,
            "verify_dps": VERIFY_DPS, "batch_size": BATCH, "flint_threads": FLINT_THREADS,
            "engine": "FLINT/Arb + Platt", "checkpoint": "GitHub" if os.getenv("CHECKPOINT_TOKEN") else "Local only",
            "last_zero": self.last, "candidate": latest("candidates"), "report": latest("reports"),
            "model": MODEL, "batch_seconds": self.batch_seconds,
        }

    async def start(self):
        if self.running:
            return
        cp = await load_checkpoint()
        saved_n = int(cp.get("next_n", START_N)) if cp else START_N
        n = max(START_N, saved_n)
        setstate("next_n", n)
        setstate("current_n", n)
        setstate("phase", "starting")
        if latest("candidates") is not None:
            self.stopped = True
            self.reason = "A candidate is already recorded; start is disabled for this run."
            return
        self.stopped = False
        self.reason = ""
        self.error = ""
        self.batches_done = 0
        self.running = True
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.running = False
        self.stopped = True
        self.reason = "Manually stopped"
        setstate("phase", "saving")
        await checkpoint(True)
        setstate("phase", "stopped")

    async def candidate(self, source, real, imag, residual, deviation, extra):
        evidence = {
            "source": source, "real_part": real, "imaginary_part": imag,
            "zeta_abs": residual, "critical_line_deviation": deviation,
            "search_precision_digits": FLINT_SEARCH_DPS, "verification_precision_digits": VERIFY_DPS,
            "extra": extra, "timestamp_utc": now(),
        }
        candidate_id = addcandidate(source, real, imag, residual, deviation, evidence)
        self.running = False
        self.stopped = True
        self.reason = "A numerically verified nontrivial zero candidate has Re(s) != 1/2"
        setstate("phase", "candidate_found")
        await checkpoint(True, {"candidate": evidence, "candidate_found": True})
        try:
            addreport(candidate_id, await asyncio.to_thread(ai_report, evidence))
        except Exception as ex:
            self.error = f"AI report generation failed: {type(ex).__name__}: {ex}"

    async def run(self):
        try:
            while self.running:
                start = max(int(state("next_n", str(START_N))), START_N)
                if MAX_N and start > MAX_N:
                    self.running = False
                    self.stopped = True
                    self.reason = f"Configured maximum n={MAX_N} reached"
                    setstate("phase", "max_reached")
                    await checkpoint(True)
                    return

                batch = min(BATCH, MAX_N - start + 1) if MAX_N else BATCH
                setstate("current_n", start)
                setstate("phase", f"computing n={start} to {start + batch - 1}")
                t0 = time.monotonic()
                zs = await asyncio.to_thread(flint_zero_batch, start, batch)
                self.batch_seconds = round(time.monotonic() - t0, 3)
                if len(zs) != batch:
                    raise RuntimeError(f"FLINT returned {len(zs)} zeros for requested batch of {batch}")

                for i, _z in enumerate(zs):
                    n = start + i
                    if not self.running:
                        return
                    setstate("phase", f"verifying n={n} at {VERIFY_DPS} digits")
                    real, imag, residual = await asyncio.to_thread(flint_verify_zero, n)
                    deviation = abs(mp.mpf(real) - mp.mpf(".5"))
                    verified = mp.mpf(residual) < mp.mpf("1e-50")
                    # zeta_zero() enumerates the critical-line sequence; off-line candidates are found separately.
                    suspicious = False
                    addzero(
                        n, real, imag, residual, mp.nstr(deviation, 60), verified, suspicious,
                        "FLINT/Arb Platt zeta-zero + 160-digit Arb verification",
                    )
                    setstate("next_n", n + 1)
                    setstate("current_n", n)
                    setstate("phase", "zero verified")
                    self.last = {
                        "n": n, "real_part": real, "imaginary_part": imag,
                        "residual_upper_bound": residual, "deviation": mp.nstr(deviation, 60),
                        "verified": verified, "batch_seconds": self.batch_seconds,
                    }
                    if not verified:
                        self.error = f"High-precision verification did not meet the residual threshold for n={n}. Search paused."
                        self.running = False
                        self.stopped = True
                        self.reason = "Verification threshold failed"
                        setstate("phase", "verification_failed")
                        await checkpoint(True, {"error": self.error})
                        return

                self.batches_done += 1
                if self.batches_done % CHECKPOINT_EVERY_BATCHES == 0:
                    setstate("phase", "saving progress")
                    await checkpoint()

                if self.batches_done % OFFLINE_EVERY_BATCHES == 0:
                    setstate("phase", "exploring off the critical line")
                    height = mp.mpf(zs[-1].imag.str(50, radius=False, more=True))
                    x = await asyncio.to_thread(exploratory, height)
                    if x is not None:
                        setstate("phase", "re-verifying possible candidate")
                        with mp.workdps(VERIFY_DPS):
                            root = mp.findroot(
                                mp.zeta, (x, x + mp.mpc(".001", ".001")),
                                solver="secant", tol=mp.mpf("1e-70"), maxsteps=80, verify=False,
                            )
                            residual2 = abs(mp.zeta(root))
                            deviation2 = abs(mp.re(root) - mp.mpf(".5"))
                        if (mp.im(root) > 10 and 0 < mp.re(root) < 1
                                and residual2 < mp.mpf("1e-50") and deviation2 > FINAL):
                            return await self.candidate(
                                "2-D off-critical-line exploratory search",
                                mp.nstr(mp.re(root), VERIFY_DPS), mp.nstr(mp.im(root), VERIFY_DPS),
                                mp.nstr(residual2, 70), mp.nstr(deviation2, 70),
                                {"seed_height": mp.nstr(height, 50), "note": "Exploratory search is not exhaustive proof."},
                            )
                await asyncio.sleep(INTERVAL)
        except asyncio.CancelledError:
            pass
        except Exception as ex:
            self.running = False
            self.stopped = True
            self.reason = "Search paused because of an execution error"
            self.error = repr(ex)
            setstate("phase", "error")
            await checkpoint(True, {"error": repr(ex)})


init()
search = Search()
app = FastAPI(title="Riemann Hypothesis Search Lab")


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


@app.get("/api/status")
async def status():
    return search.status()


@app.post("/api/start")
async def start():
    await search.start()
    return search.status()


@app.post("/api/stop")
async def stop():
    await search.stop()
    return search.status()


@app.get("/api/zeros")
async def api_zeros(limit: int = 30):
    return zeros(min(max(limit, 1), 500))


@app.get("/api/candidate")
async def api_candidate():
    return latest("candidates")


@app.get("/api/report")
async def api_report():
    return latest("reports")


@app.get("/api/health")
async def health():
    return {
        "ok": True, "running": search.running, "start_n": START_N,
        "next_n": int(state("next_n")), "current_n": int(state("current_n", state("next_n"))),
        "phase": state("phase", "idle"),
        "checkpoint": "github" if os.getenv("CHECKPOINT_TOKEN") else "local-only",
        "engine": "FLINT/Arb + Platt", "flint_threads": FLINT_THREADS,
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Riemann Hypothesis Search</title>
<style>
:root{--ink:#171717;--muted:#5f6368;--paper:#fbfaf6;--card:#fff;--line:#222;--soft:#f0eee8}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 18px 60px} header{margin-bottom:22px}
h1{font-family:Georgia,serif;font-size:clamp(30px,5vw,52px);line-height:1.05;margin:0 0 8px}
.subtitle{font-size:17px;color:var(--muted);max-width:900px;margin:0;line-height:1.5}
.actions{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0} button{font:inherit;font-weight:750;border:2px solid var(--line);border-radius:10px;padding:12px 18px;background:#fff;cursor:pointer}
button.primary{background:var(--ink);color:#fff} button:hover{transform:translateY(-1px)}
.section-title{font-size:14px;text-transform:uppercase;letter-spacing:.09em;font-weight:800;margin:25px 0 9px;color:var(--muted)}
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:var(--card);border:2px solid var(--line);border-radius:13px;padding:16px;box-shadow:3px 3px 0 #d7d4cc}
.label{font-size:14px;font-weight:800;color:var(--muted);margin-bottom:5px}.value{font-size:25px;font-weight:850;word-break:break-word}.small{font-size:13px;color:var(--muted);margin-top:5px;line-height:1.4}
.row{display:grid;grid-template-columns:220px 1fr;align-items:center;gap:14px;background:var(--card);border:2px solid var(--line);border-radius:10px;padding:13px 16px;margin:9px 0}
.row-label{font-weight:850;font-size:17px}.row-value{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:17px;overflow:auto;white-space:nowrap}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:var(--card);border:2px solid var(--line);border-radius:13px;padding:18px;margin-top:12px}
.panel h2{font-size:20px;margin:0 0 7px}.explain{line-height:1.55;color:#30343a}.note{font-size:13px;color:var(--muted);line-height:1.5}
pre{margin:10px 0 0;white-space:pre-wrap;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.45;background:var(--soft);border-radius:8px;padding:12px}
.footer{margin-top:24px;color:var(--muted);font-size:12px;line-height:1.5}
@media(max-width:760px){.status-grid{grid-template-columns:1fr 1fr}.row{grid-template-columns:1fr;gap:5px}.two{grid-template-columns:1fr}}
@media(max-width:470px){.status-grid{grid-template-columns:1fr}.value{font-size:22px}}
</style></head>
<body><main>
<header><h1>Riemann Hypothesis Search</h1>
<p class="subtitle">A continuous numerical investigation of the nontrivial zeros of the Riemann zeta function. Everything below is written so you can follow the search without knowing how the software works.</p></header>

<div class="actions"><button class="primary" onclick="startSearch()">▶ Start / Resume</button><button onclick="stopSearch()">■ Stop &amp; Save</button></div>

<div class="section-title">Search status</div>
<div class="status-grid">
<div class="card"><div class="label">Status</div><div id="status" class="value">—</div><div id="reason" class="small">—</div></div>
<div class="card"><div class="label">Next zero number (n)</div><div id="next" class="value">—</div><div class="small">The next zero the program will examine.</div></div>
<div class="card"><div class="label">Current zero number (n)</div><div id="current" class="value">—</div><div class="small">The zero currently being processed.</div></div>
<div class="card"><div class="label">Zeros fully checked</div><div id="checked" class="value">—</div><div class="small">Zeros whose high-precision verification finished.</div></div>
<div class="card"><div class="label">What it is doing</div><div id="phase" class="value">—</div><div class="small">Plain-language description of the current step.</div></div>
<div class="card"><div class="label">Verification precision</div><div id="precision" class="value">—</div><div class="small">Decimal digits used for final numerical verification.</div></div>
</div>

<div class="section-title">Latest verified zero</div>
<div class="row"><div class="row-label">n</div><div id="zn" class="row-value">—</div></div>
<div class="row"><div class="row-label">Real part</div><div id="zr" class="row-value">—</div></div>
<div class="row"><div class="row-label">Imaginary part</div><div id="zi" class="row-value">—</div></div>
<div class="row"><div class="row-label">Residual |ζ(s)|</div><div id="zres" class="row-value">—</div></div>
<div class="row"><div class="row-label">Distance from 1/2</div><div id="zdev" class="row-value">—</div></div>

<div class="section-title">How the search is running</div>
<div class="two">
<div class="panel"><h2>⚡ Fast numerical engine</h2>
<p class="explain">The search uses <b>FLINT/Arb</b> and its high-index <b>Platt</b> zeta-zero algorithms. Zeros are requested in batches to reduce overhead, then each zero is checked again at high precision.</p>
<div class="small">Engine: <b id="engine">—</b> · Batch size: <b id="batch">—</b> · Internal threads: <b id="threads">—</b></div></div>
<div class="panel"><h2>💾 Progress protection</h2>
<p class="explain">Progress is periodically saved to the configured GitHub checkpoint. If the hosting service restarts, the search can resume from the saved zero number instead of starting over.</p>
<div class="small">Checkpoint: <b id="checkpoint">—</b> · Last batch time: <b id="batchtime">—</b></div></div>
</div>

<div class="section-title">If something unusual is found</div>
<div class="panel"><h2>🔎 Possible off-critical-line candidate</h2>
<p class="explain">The normal zero enumeration is for the known sequence on the critical line. Separately, the program periodically performs an exploratory two-dimensional search away from that line. If that exploration finds a candidate that survives the stronger numerical check, the search stops and the evidence is saved.</p>
<p class="note">Important: an exploratory numerical search is not a formal proof that no other counterexample exists, and the AI report is an assistant's explanation of the evidence—not a mathematical authority.</p>
<pre id="candidate">None</pre></div>

<div class="section-title">Research report</div>
<div class="panel"><h2>🤖 AI research assistant</h2><div class="small">Model: <b id="model">—</b></div><pre id="report">No report has been generated.</pre></div>

<div class="section-title">Technical details</div>
<div class="panel"><h2>For anyone who wants the deeper details</h2><pre id="details">—</pre></div>

<div class="section-title">Messages</div><div class="panel"><pre id="error">No errors.</pre></div>
<div class="footer">This application performs numerical computation. Numerical evidence, even at high precision, should not be presented as a formal proof of the Riemann Hypothesis.</div>
</main>
<script>
const $=id=>document.getElementById(id);
function fmt(v){return v===null||v===undefined?'—':String(v)}
function shortPhase(p){if(!p)return'Idle';if(p.startsWith('computing'))return'Finding zeros';if(p.startsWith('verifying'))return'Verifying a zero';if(p==='zero verified')return'Zero verified';if(p==='saving progress')return'Saving progress';if(p==='exploring off the critical line')return'Exploring for a possible counterexample';if(p==='re-verifying possible candidate')return'Re-checking a possible candidate';if(p==='candidate_found')return'Candidate found — search stopped';if(p==='verification_failed')return'Verification failed — search stopped';if(p==='max_reached')return'Configured limit reached';if(p==='error')return'Error — search stopped';if(p==='stopped')return'Stopped and saved';if(p==='starting')return'Starting';return p}
function render(x){
$('status').textContent=x.running?'RUNNING':(x.stopped?'STOPPED':'PAUSED');$('reason').textContent=x.reason||'Search is ready.';
$('next').textContent=fmt(x.next_n);$('current').textContent=fmt(x.current_n);$('checked').textContent=fmt(x.zeros_checked);$('phase').textContent=shortPhase(x.phase);$('precision').textContent=fmt(x.verify_dps)+' digits';
const z=x.last_zero;$('zn').textContent=z?fmt(z.n):'—';$('zr').textContent=z?fmt(z.real_part):'—';$('zi').textContent=z?fmt(z.imaginary_part):'—';$('zres').textContent=z?fmt(z.residual_upper_bound):'—';$('zdev').textContent=z?fmt(z.deviation):'—';
$('engine').textContent=fmt(x.engine);$('batch').textContent=fmt(x.batch_size);$('threads').textContent=fmt(x.flint_threads);$('checkpoint').textContent=fmt(x.checkpoint);$('batchtime').textContent=x.batch_seconds==null?'—':x.batch_seconds+' s';$('model').textContent=fmt(x.model);
$('candidate').textContent=x.candidate?JSON.stringify(x.candidate,null,2):'None';$('report').textContent=x.report?x.report.report:'No report has been generated.';
$('details').textContent=JSON.stringify({search_start_n:x.start_n,next_n:x.next_n,current_n:x.current_n,zeros_checked:x.zeros_checked,search_precision_digits:x.search_dps,verification_precision_digits:x.verify_dps,batch_size:x.batch_size,flint_threads:x.flint_threads,engine:x.engine,checkpoint:x.checkpoint,phase:x.phase},null,2);
$('error').textContent=x.error||'No errors.'}
async function startSearch(){try{render(await(await fetch('/api/start',{method:'POST'})).json())}catch(e){$('error').textContent=String(e)}}
async function stopSearch(){try{render(await(await fetch('/api/stop',{method:'POST'})).json())}catch(e){$('error').textContent=String(e)}}
async function poll(){try{render(await(await fetch('/api/status',{cache:'no-store'})).json())}catch(e){$('error').textContent='Dashboard temporarily could not reach the search service.'}}
poll();setInterval(poll,1000);
</script></body></html>'''
