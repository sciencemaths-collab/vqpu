"""vQPU Console — a self-contained web UI for the quantum VM.

One process, one port. Opens a browser tab with:

    ┌─────────────┬─────────────────────────────────┬───────────────────┐
    │ Backends    │  >>> terminal (persistent REPL) │ Artifacts         │
    │  CPU   ●    │  ────────────────────────────── │  counts histogram │
    │  GPU   ○    │  [multi-line editor]            │  last seq (len/2Q)│
    │  IonQ  ●    │  ──────────────── [Run] [Reset] │  fidelity / depth │
    └─────────────┴─────────────────────────────────┴───────────────────┘

The REPL is a `code.InteractiveInterpreter` with vqpu / chesso / AEGIS-Ion-N
already imported. Anything you type is executed against the *same* Python
session for the lifetime of the console — variables persist, circuits
persist, state vectors persist. The artifacts panel auto-updates whenever
you assign a dict of counts to ``counts``, a gate_sequence to ``seq``, or
set ``n_qubits``.

Run:
    ./.venv/bin/python vqpu_console.py
    → opens http://127.0.0.1:7777

Use env vars to preload IonQ credentials into the session so your code can
do ``QPUCloudPlugin("ionq").execute_sample(...)`` without extra plumbing:
    IONQ_API_KEY=<key> IONQ_BACKEND=ionq_simulator \
        ./.venv/bin/python vqpu_console.py
"""

from __future__ import annotations

import code
import contextlib
import io
import json
import os
import pathlib
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flask import Flask, jsonify, request

from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q


# ─────────────────────────── persistent REPL ──────────────────────────────

_PRELOAD = """
import numpy as np
import vqpu
from vqpu import (
    QPUCloudPlugin, UniversalvQPU, CPUPlugin,
    QuantumCircuit, QuantumRegister, vQPU, QuantumAlgorithms,
)
from vqpu.chesso import compile_qlambda_for_hardware, execute_qlambda_on_backend
from vqpu.chesso.experiments import aegis_ion_nested
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.chesso.experiments.ionq_noise import (
    IonQNoiseSpec, sample_with_ionq_noise, ideal_counts, expected_circuit_fidelity,
)

# Artifact hooks — assign these to see them in the right-hand panel.
counts = None
seq = None
n_qubits = None
note = None
"""


class ReplSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.history: List[Dict[str, str]] = []
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.interp = code.InteractiveInterpreter()
        self.interp.runsource("import builtins", "<preload>", "exec")
        # Execute the preload block line-by-line so compile errors surface.
        for block in _PRELOAD.strip().split("\n\n"):
            self._run_block(block, record=False)

    def reset(self) -> None:
        with self._lock:
            self.history.clear()
            self._reset_locked()

    def _run_block(self, source: str, *, record: bool) -> Dict[str, Any]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        ok = True
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # Try as an expression first so bare values print their repr.
                compiled = None
                try:
                    compiled = compile(source, "<repl>", "eval")
                    result = eval(compiled, self.interp.locals)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    compiled = compile(source, "<repl>", "exec")
                    exec(compiled, self.interp.locals)
            except SystemExit:
                print("(SystemExit suppressed in console)", file=stderr)
                ok = False
            except BaseException:
                traceback.print_exc(file=stderr)
                ok = False
        entry = {
            "source": source,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "ok": ok,
        }
        if record:
            self.history.append(entry)
        return entry

    def run(self, source: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._run_block(source, record=True)
            entry["artifacts"] = self._snapshot_artifacts()
            return entry

    def _snapshot_artifacts(self) -> Dict[str, Any]:
        ns = self.interp.locals
        counts = ns.get("counts")
        seq = ns.get("seq")
        n_qubits = ns.get("n_qubits")
        note = ns.get("note")

        def _counts_payload(c: Any) -> Optional[Dict[str, int]]:
            if not isinstance(c, dict):
                return None
            out: Dict[str, int] = {}
            for k, v in c.items():
                try:
                    out[str(k)] = int(v)
                except Exception:
                    return None
            return out

        def _seq_payload(s: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(s, list) or not s:
                return None
            head = []
            for g in s[:12]:
                if not isinstance(g, (list, tuple)) or len(g) < 2:
                    continue
                name = str(g[0])
                raw_t = g[1]
                targets = list(raw_t) if isinstance(raw_t, (list, tuple)) else [raw_t]
                params = []
                for p in list(g[2:]):
                    if hasattr(p, "shape"):
                        params.append(f"<{type(p).__name__}{tuple(p.shape)}>")
                    elif isinstance(p, float):
                        params.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        params.append(str(p))
                    else:
                        params.append(type(p).__name__)
                head.append({"name": name, "targets": [int(t) for t in targets], "params": params})
            try:
                two_q = count_2q(s)
            except Exception:
                two_q = 0
            depth = None
            if isinstance(n_qubits, int) and n_qubits > 0:
                try:
                    depth = circuit_depth(s, int(n_qubits))
                except Exception:
                    depth = None
            return {
                "length": len(s),
                "two_q": two_q,
                "depth": depth,
                "head": head,
                "truncated": len(s) > 12,
            }

        return {
            "counts": _counts_payload(counts),
            "seq": _seq_payload(seq),
            "n_qubits": int(n_qubits) if isinstance(n_qubits, int) else None,
            "note": str(note) if note is not None else None,
        }


SESSION = ReplSession()


# ─────────────────────────── backend probe ────────────────────────────────

def probe_backends() -> List[Dict[str, Any]]:
    from vqpu import UniversalvQPU
    try:
        vqpu = UniversalvQPU(verbose=False)
    except Exception as exc:
        return [{"name": "probe-failed", "ok": False, "error": str(exc)}]
    out: List[Dict[str, Any]] = []
    for name, fp in vqpu.backends.items():
        out.append({
            "name": fp.name,
            "provider": name,
            "is_available": bool(fp.is_available),
            "is_local": bool(fp.is_local),
            "max_qubits": int(fp.max_qubits),
            "compute_class": getattr(fp.compute_class, "name", str(fp.compute_class)),
            "native_gates": list(fp.native_gates),
            "best_for": list(fp.best_for),
        })
    return out


# ────────────────────────── quick-run snippets ────────────────────────────

QUICK_SNIPPETS = [
    {
        "title": "GHZ-5 on CPU",
        "code": (
            "qpu = CPUPlugin()\n"
            "n_qubits = 5\n"
            "seq = [(\"H\",[0])] + [(\"CNOT\",[i,i+1]) for i in range(4)]\n"
            "counts = qpu.execute_sample(n_qubits, seq, shots=1024)\n"
            "note = \"GHZ-5 on local CPU statevector\""
        ),
    },
    {
        "title": "AEGIS on redundant GHZ-5",
        "code": (
            "src = \"program redundant_ghz\\nalloc q0\\nalloc q1\\nalloc q2\\nalloc q3\\nalloc q4\\n\"\n"
            "src += \"gate H q0\\ngate H q0\\ngate H q0\\n\"\n"
            "for i in range(1,5):\n"
            "    src += f\"gate CX q{i-1} q{i}\\ngate CX q{i-1} q{i}\\ngate CX q{i-1} q{i}\\n\"\n"
            "bridged = compile_qlambda_for_hardware(src)\n"
            "baseline = list(bridged.gate_sequence)\n"
            "n_qubits = bridged.n_qubits\n"
            "res = aegis_ion_nested(baseline, n_qubits)\n"
            "seq = res.winner.sequence\n"
            "counts = CPUPlugin().execute_sample(n_qubits, seq, 1024)\n"
            "note = f\"winner={res.winner.strategy}  2Q: {count_2q(baseline)}->{count_2q(seq)}\""
        ),
    },
    {
        "title": "Submit GHZ-3 to IonQ",
        "code": (
            "# requires IONQ_API_KEY with jobs:write scope\n"
            "ionq = QPUCloudPlugin(\"ionq\")\n"
            "n_qubits = 3\n"
            "seq = [(\"H\",[0]),(\"CNOT\",[0,1]),(\"CNOT\",[1,2])]\n"
            "counts = ionq.execute_sample(n_qubits=n_qubits, gate_sequence=seq, shots=256)\n"
            "note = \"GHZ-3 on IonQ (backend from IONQ_BACKEND env)\""
        ),
    },
    {
        "title": "Aria-spec noise preview",
        "code": (
            "n_qubits = 5\n"
            "seq = [(\"H\",[0])] + [(\"CNOT\",[i,i+1]) for i in range(4)]\n"
            "counts = sample_with_ionq_noise(n_qubits, seq, 500, spec=IonQNoiseSpec.aria(), seed=1)\n"
            "note = f\"expected F = {expected_circuit_fidelity(seq):.4f} at Aria noise\""
        ),
    },
]


# ─────────────────────────── Flask routes ─────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return INDEX_HTML


@app.route("/api/backends")
def api_backends():
    return jsonify(probe_backends())


@app.route("/api/snippets")
def api_snippets():
    return jsonify(QUICK_SNIPPETS)


@app.route("/api/env")
def api_env():
    ionq_key = os.environ.get("IONQ_API_KEY")
    return jsonify({
        "ionq_key_set": bool(ionq_key),
        "ionq_key_len": len(ionq_key) if ionq_key else 0,
        "ionq_backend": os.environ.get("IONQ_BACKEND", ""),
        "ionq_noise_model": os.environ.get("IONQ_NOISE_MODEL", ""),
        "python": sys.version.split()[0],
        "cwd": str(pathlib.Path.cwd()),
    })


@app.route("/api/exec", methods=["POST"])
def api_exec():
    data = request.get_json(silent=True) or {}
    source = str(data.get("code", ""))
    if not source.strip():
        return jsonify({"ok": False, "stderr": "(empty input)", "stdout": "", "artifacts": None})
    entry = SESSION.run(source)
    return jsonify(entry)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    SESSION.reset()
    return jsonify({"ok": True})


@app.route("/api/history")
def api_history():
    return jsonify(SESSION.history[-50:])


# ─────────────────────────── HTML / JS / CSS ──────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>vQPU Console</title>
<style>
  :root {
    --bg:        #0b0e14;
    --panel:     #12161f;
    --panel-2:   #181c26;
    --border:    #242a37;
    --fg:        #d4d8e0;
    --fg-dim:    #7d8798;
    --accent:    #7ab6ff;
    --accent-2:  #5ce6b4;
    --warn:      #ffb454;
    --err:       #ff6060;
    --good:      #5ce6b4;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--fg);
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }
  header {
    padding: 10px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 14px;
  }
  header .logo { font-weight: 700; color: var(--accent); letter-spacing: 0.04em; }
  header .sub { color: var(--fg-dim); }
  header .spacer { flex: 1; }
  header button {
    background: var(--panel-2); color: var(--fg);
    border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-size: 12px;
  }
  header button:hover { border-color: var(--accent); color: var(--accent); }
  main {
    display: grid;
    grid-template-columns: 240px 1fr 320px;
    height: calc(100vh - 45px);
  }
  .pane {
    border-right: 1px solid var(--border);
    background: var(--panel);
    overflow-y: auto;
  }
  .pane:last-child { border-right: none; border-left: 1px solid var(--border); }
  .pane h3 {
    margin: 0; padding: 10px 14px;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--fg-dim); border-bottom: 1px solid var(--border);
  }
  .pane .section { padding: 10px 14px; border-bottom: 1px solid var(--border); }

  .backend-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0; font-size: 12px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
  .dot.on  { background: var(--good); box-shadow: 0 0 4px var(--good); }
  .dot.off { background: var(--fg-dim); }
  .backend-meta { color: var(--fg-dim); font-size: 11px; }

  .term {
    display: flex; flex-direction: column; height: 100%;
    background: var(--bg);
  }
  .term .log {
    flex: 1; overflow-y: auto; padding: 14px;
    white-space: pre-wrap; word-break: break-word;
  }
  .log .entry { margin-bottom: 10px; }
  .log .prompt { color: var(--accent); }
  .log .src { color: var(--fg); }
  .log .out { color: var(--fg); margin-top: 4px; }
  .log .err { color: var(--err); margin-top: 4px; }
  .term .input {
    display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--border);
    background: var(--panel);
  }
  .term textarea {
    flex: 1; background: var(--panel-2); color: var(--fg); border: 1px solid var(--border);
    padding: 8px 10px; border-radius: 4px; font-family: inherit; font-size: 13px;
    min-height: 42px; resize: vertical;
  }
  .term .btns { display: flex; flex-direction: column; gap: 6px; }
  .term button {
    padding: 4px 14px; background: var(--accent); color: #0b0e14;
    border: none; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-weight: 600; font-size: 12px;
  }
  .term button.secondary { background: var(--panel-2); color: var(--fg); border: 1px solid var(--border); }

  .chip {
    display: inline-block; padding: 2px 7px; border-radius: 3px;
    background: var(--panel-2); color: var(--fg-dim); font-size: 10px;
    border: 1px solid var(--border); margin-right: 4px;
  }
  .chip.good { color: var(--good); border-color: var(--good); }
  .chip.warn { color: var(--warn); border-color: var(--warn); }

  .hist-bar {
    display: flex; align-items: center; font-size: 11px;
    margin: 2px 0; gap: 6px;
  }
  .hist-bar .lbl { color: var(--fg-dim); min-width: 75px; text-align: right; font-family: inherit; }
  .hist-bar .bar-outer { flex: 1; background: var(--panel-2); border-radius: 3px; height: 12px; overflow: hidden; }
  .hist-bar .bar-inner { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
  .hist-bar .v { color: var(--fg); min-width: 40px; }

  .snippet {
    display: block; background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 4px; padding: 7px 10px; margin: 6px 0;
    cursor: pointer; color: var(--fg); font-size: 12px;
  }
  .snippet:hover { border-color: var(--accent); color: var(--accent); }

  .kv { display: grid; grid-template-columns: 95px 1fr; gap: 4px 8px; font-size: 12px; }
  .kv .k { color: var(--fg-dim); }
  .kv .v { color: var(--fg); }

  pre.seqhead {
    background: var(--panel-2); border: 1px solid var(--border); padding: 6px 8px;
    border-radius: 4px; font-size: 11px; overflow-x: auto;
    color: var(--fg-dim); max-height: 180px;
  }
  pre.seqhead .gname { color: var(--accent-2); }

  footer {
    position: fixed; bottom: 0; right: 0; padding: 3px 10px;
    font-size: 11px; color: var(--fg-dim);
  }

  @media (max-width: 1100px) {
    main { grid-template-columns: 200px 1fr 280px; }
  }
</style>
</head>
<body>
<header>
  <span class="logo">vQPU</span>
  <span class="sub">Quantum VM console</span>
  <span class="spacer"></span>
  <span id="envChip" class="chip">env</span>
  <button id="btnReset">reset session</button>
  <button id="btnProbe">re-probe</button>
</header>
<main>
  <aside class="pane">
    <h3>Backends</h3>
    <div id="backends" class="section"><em class="backend-meta">probing…</em></div>
    <h3>Quick runs</h3>
    <div id="snippets" class="section"></div>
  </aside>

  <section class="term">
    <div id="log" class="log"></div>
    <div class="input">
      <textarea id="codebox" placeholder="Python here.  Shift+Enter = newline, Cmd/Ctrl+Enter = run.&#10;vqpu / chesso / AEGIS already imported."></textarea>
      <div class="btns">
        <button id="btnRun">▶ Run</button>
        <button id="btnClear" class="secondary">clear log</button>
      </div>
    </div>
  </section>

  <aside class="pane">
    <h3>Artifacts</h3>
    <div id="artifacts" class="section"><em class="backend-meta">no artifacts yet — assign <code>counts</code>, <code>seq</code>, <code>n_qubits</code> in the terminal.</em></div>
  </aside>
</main>
<footer id="footer">ready.</footer>

<script>
const $ = (id) => document.getElementById(id);
const log = $("log");

function appendEntry(entry) {
  const div = document.createElement("div");
  div.className = "entry";
  const src = document.createElement("div");
  src.innerHTML = '<span class="prompt">&gt;&gt;&gt; </span><span class="src"></span>';
  src.querySelector(".src").textContent = entry.source;
  div.appendChild(src);
  if (entry.stdout) {
    const out = document.createElement("div"); out.className = "out";
    out.textContent = entry.stdout.replace(/\n$/, "");
    div.appendChild(out);
  }
  if (entry.stderr) {
    const err = document.createElement("div"); err.className = "err";
    err.textContent = entry.stderr.replace(/\n$/, "");
    div.appendChild(err);
  }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderArtifacts(a) {
  const el = $("artifacts");
  if (!a || (!a.counts && !a.seq && a.n_qubits == null && !a.note)) return;
  el.innerHTML = "";
  if (a.note) {
    const n = document.createElement("div");
    n.style.color = "var(--accent-2)";
    n.style.marginBottom = "8px";
    n.textContent = a.note;
    el.appendChild(n);
  }
  if (a.seq) {
    const kv = document.createElement("div"); kv.className = "kv";
    kv.innerHTML =
      `<span class="k">length</span><span class="v">${a.seq.length}</span>` +
      `<span class="k">2Q count</span><span class="v">${a.seq.two_q}</span>` +
      `<span class="k">depth</span><span class="v">${a.seq.depth ?? "(set n_qubits)"}</span>` +
      `<span class="k">qubits</span><span class="v">${a.n_qubits ?? "?"}</span>`;
    el.appendChild(kv);
    const pre = document.createElement("pre"); pre.className = "seqhead";
    const lines = a.seq.head.map(g => {
      const p = g.params.length ? " " + g.params.join(",") : "";
      return `<span class="gname">${g.name}</span>(${g.targets.join(",")}${p})`;
    });
    if (a.seq.truncated) lines.push(`… (+${a.seq.length - a.seq.head.length} more)`);
    pre.innerHTML = lines.join("\n");
    el.appendChild(pre);
  }
  if (a.counts) {
    const entries = Object.entries(a.counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    const hdr = document.createElement("div");
    hdr.style.margin = "10px 0 4px 0";
    hdr.style.color = "var(--fg-dim)";
    hdr.textContent = `counts (top ${entries.length} of ${Object.keys(a.counts).length})`;
    el.appendChild(hdr);
    for (const [bits, v] of entries) {
      const row = document.createElement("div"); row.className = "hist-bar";
      const pct = Math.max(1, Math.round(100 * v / total));
      row.innerHTML =
        `<span class="lbl">${bits}</span>` +
        `<span class="bar-outer"><span class="bar-inner" style="width:${pct}%"></span></span>` +
        `<span class="v">${v}</span>`;
      el.appendChild(row);
    }
  }
}

function renderBackends(list) {
  const host = $("backends");
  host.innerHTML = "";
  if (!list.length) { host.innerHTML = "<em>(none)</em>"; return; }
  for (const b of list) {
    const row = document.createElement("div"); row.className = "backend-row";
    row.innerHTML =
      `<span><span class="dot ${b.is_available ? 'on' : 'off'}"></span>${b.name}</span>` +
      `<span class="backend-meta">${b.max_qubits}q · ${b.is_local ? 'local' : 'cloud'}</span>`;
    host.appendChild(row);
    const sub = document.createElement("div"); sub.className = "backend-meta";
    sub.style.marginBottom = "6px";
    sub.textContent = b.compute_class + " · " + (b.native_gates.slice(0, 6).join(",") || "");
    host.appendChild(sub);
  }
}

function renderSnippets(list) {
  const host = $("snippets"); host.innerHTML = "";
  for (const s of list) {
    const a = document.createElement("span"); a.className = "snippet";
    a.textContent = s.title;
    a.onclick = () => { $("codebox").value = s.code; $("codebox").focus(); };
    host.appendChild(a);
  }
}

async function exec() {
  const code = $("codebox").value;
  if (!code.trim()) return;
  $("footer").textContent = "running…";
  const r = await fetch("/api/exec", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ code })
  });
  const entry = await r.json();
  appendEntry(entry);
  if (entry.artifacts) renderArtifacts(entry.artifacts);
  $("codebox").value = "";
  $("codebox").focus();
  $("footer").textContent = entry.ok ? "ok" : "error";
}

async function reset() {
  await fetch("/api/reset", { method: "POST" });
  log.innerHTML = "";
  $("artifacts").innerHTML = '<em class="backend-meta">session reset.</em>';
  $("footer").textContent = "session reset";
}

async function probe() {
  $("backends").innerHTML = '<em class="backend-meta">probing…</em>';
  const r = await fetch("/api/backends");
  renderBackends(await r.json());
}

async function envchip() {
  const r = await fetch("/api/env");
  const e = await r.json();
  let s = `py ${e.python}`;
  if (e.ionq_key_set) s += ` · IonQ key(${e.ionq_key_len})`;
  if (e.ionq_backend) s += ` · ${e.ionq_backend}`;
  if (e.ionq_noise_model) s += ` · noise=${e.ionq_noise_model}`;
  $("envChip").textContent = s;
  if (e.ionq_key_set) $("envChip").classList.add("good");
}

$("btnRun").onclick = exec;
$("btnReset").onclick = reset;
$("btnClear").onclick = () => { log.innerHTML = ""; };
$("btnProbe").onclick = probe;
$("codebox").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); exec(); }
});

probe(); envchip();
fetch("/api/snippets").then(r => r.json()).then(renderSnippets);
</script>
</body>
</html>
"""


# ──────────────────────────────── main ────────────────────────────────────

def main() -> int:
    host = os.environ.get("VQPU_CONSOLE_HOST", "127.0.0.1")
    port = int(os.environ.get("VQPU_CONSOLE_PORT", "7777"))
    url = f"http://{host}:{port}"
    if os.environ.get("VQPU_CONSOLE_OPEN", "1") == "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    print(f"vQPU console → {url}")
    print("  press Ctrl+C to stop.")
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
