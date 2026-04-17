"""vQPU Desktop — VM-style shell for the quantum virtual machine.

What you see when you launch
────────────────────────────
A full-screen dark desktop with a grid of app icons, a top menu bar, and
a bottom taskbar. Double-click any icon (or click it once) to open that
app as its own window. Multiple apps can be open at once; the taskbar
shows every open window and lets you raise/minimize it. The bottom-right
system tray shows which quantum/classical backends are linked right now,
heartbeat latency, and the date.

Apps
────
    Terminal          persistent Python REPL with vqpu/chesso/AEGIS
                      pre-imported; Enter runs, ⌘O loads a .py file
    Link Manager      forge/close IonQ links; see health, latency, calls
    Backends          discovery panel — CPU/GPU/TPU/QPU probed live
    Workloads         one-click templates (GHZ, AEGIS, TSP, noise preview…)
    Job Monitor       live stream of recent submissions through LinkManager
    Circuit Viewer    last `seq` assigned in the REPL, rendered as text
    Docs              opens IonQ console / vqpu docs in your system browser

Shortcuts
─────────
    ⌘↵ / Ctrl↵        run current code in the Terminal
    ⌘O                open a .py file into the Terminal
    ⌘L                clear the REPL log
    Esc               close the active dialog
"""

from __future__ import annotations

import code
import contextlib
import io
import os
import pathlib
import queue
import sys
import textwrap
import threading
import time
import traceback
import webbrowser
import tkinter as tk
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from vqpu import (
    CPUPlugin, NvidiaGPUPlugin, AMDGPUPlugin, IntelGPUPlugin,
    AppleSiliconPlugin, TPUPlugin, QPUCloudPlugin,
)
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.link import (
    LinkManager, LinkState, QuantumTask,
    CloudQuantumLink, LocalQuantumLink, QuantumLink,
)


# ═══════════════════════════ theme ════════════════════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

T = {
    "bg":        "#060810",   # desktop wallpaper (near-black with blue tint)
    "bg2":       "#0a0e18",
    "panel":     "#0f131e",   # app window background
    "panel2":    "#151a28",
    "panel3":    "#1c2233",
    "border":    "#262d41",
    "fg":        "#e6e9f0",
    "fg2":       "#a8aec0",
    "fg_dim":    "#646b80",
    "accent":    "#5fa8ff",
    "accent2":   "#4dd5b0",
    "warn":      "#ffb454",
    "err":       "#ff5c5c",
    "good":      "#4dd5b0",
    "muted":     "#384154",
    "taskbar":   "#0a0e18",
    "font":      ("SF Mono", 13),
    "font_sm":   ("SF Mono", 11),
    "font_xs":   ("SF Mono", 10),
    "font_ui":   ("SF Pro Text", 12),
    "font_hdr":  ("SF Pro Display", 13, "bold"),
    "font_big":  ("SF Pro Display", 20, "bold"),
    "font_icon": ("SF Pro Display", 26, "bold"),
    "font_label":("SF Pro Text", 11),
}

CORNER = 12
BTN_CORNER = 8


# ═══════════════════════════ REPL engine ══════════════════════════════════

_PRELOAD = """
import numpy as np
import vqpu
from vqpu import (
    QPUCloudPlugin, UniversalvQPU, CPUPlugin,
    QuantumCircuit, QuantumRegister, vQPU,
)
from vqpu.chesso import compile_qlambda_for_hardware, execute_qlambda_on_backend
from vqpu.chesso.experiments import aegis_ion_nested
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.chesso.experiments.ionq_noise import (
    IonQNoiseSpec, sample_with_ionq_noise, ideal_counts, expected_circuit_fidelity,
)
from vqpu.link import LinkManager, QuantumTask, LinkState

counts = None
seq = None
n_qubits = None
note = None
"""


class ReplEngine:
    """Persistent Python session executed on a worker thread."""

    def __init__(self, link_manager: LinkManager) -> None:
        self._lock = threading.Lock()
        self._jobs: "queue.Queue[tuple[str, queue.Queue]]" = queue.Queue()
        self._link_manager = link_manager
        # Job log: every _exec appends a record here. Read by Job Monitor.
        self.job_log: List[Dict[str, Any]] = []
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._reset_locked()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _reset_locked(self) -> None:
        self.interp = code.InteractiveInterpreter()
        self.interp.runsource("import builtins", "<preload>", "exec")
        self.interp.runsource(_PRELOAD, "<preload>", "exec")
        self.interp.locals["links"] = self._link_manager

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def submit(self, source: str) -> "queue.Queue[dict]":
        out_q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self._jobs.put((source, out_q))
        return out_q

    def subscribe(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    def _run(self) -> None:
        while True:
            source, out_q = self._jobs.get()
            entry = self._exec(source)
            out_q.put(entry)
            self.job_log.append(entry)
            if len(self.job_log) > 200:
                self.job_log.pop(0)
            for fn in list(self._listeners):
                try:
                    fn(entry)
                except Exception:
                    pass

    def _exec(self, source: str) -> Dict[str, Any]:
        stdout, stderr = io.StringIO(), io.StringIO()
        ok = True
        t0 = time.perf_counter()
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        normalized = textwrap.dedent(normalized)
        normalized = "\n".join(ln.rstrip() for ln in normalized.split("\n"))
        src = normalized
        with self._lock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                try:
                    compiled = compile(src, "<repl>", "eval")
                    result = eval(compiled, self.interp.locals)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    try:
                        compiled = compile(src, "<repl>", "exec")
                    except IndentationError:
                        repaired = "\n".join(
                            (ln if not ln.strip() else ln.lstrip())
                            for ln in src.split("\n")
                        )
                        compiled = compile(repaired, "<repl>", "exec")
                        print("(auto-repaired leading whitespace)", file=stderr)
                    exec(compiled, self.interp.locals)
            except SystemExit:
                ok = False
                print("(SystemExit suppressed)", file=stderr)
            except BaseException:
                ok = False
                traceback.print_exc(file=stderr)
        return {
            "source": source,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "ok": ok,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "t": time.time(),
            "artifacts": self._artifacts(),
        }

    def _artifacts(self) -> Dict[str, Any]:
        ns = self.interp.locals
        counts, seq, n_qubits, note = (
            ns.get("counts"), ns.get("seq"), ns.get("n_qubits"), ns.get("note"),
        )
        art: Dict[str, Any] = {
            "note": str(note) if note is not None else None,
            "n_qubits": int(n_qubits) if isinstance(n_qubits, int) else None,
        }
        if isinstance(counts, dict):
            try:
                art["counts"] = {str(k): int(v) for k, v in counts.items()}
            except Exception:
                art["counts"] = None
        else:
            art["counts"] = None
        if isinstance(seq, list) and seq:
            try:
                two_q = count_2q(seq)
            except Exception:
                two_q = 0
            depth = None
            if art["n_qubits"]:
                try:
                    depth = circuit_depth(seq, art["n_qubits"])
                except Exception:
                    depth = None
            head_lines: List[str] = []
            for g in seq[:14]:
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
                head_lines.append(
                    f"{name}({','.join(str(int(t)) for t in targets)}"
                    f"{', ' + ', '.join(params) if params else ''})"
                )
            more = f"\n… (+{len(seq)-len(head_lines)} more)" if len(seq) > len(head_lines) else ""
            art["seq"] = {
                "length": len(seq), "two_q": two_q, "depth": depth,
                "head_text": "\n".join(head_lines) + more,
            }
        else:
            art["seq"] = None
        return art


# ═════════════════════ discovery + templates ══════════════════════════════

_DISCOVERABLE = [
    ("CPU",          "Classical CPU",       lambda: CPUPlugin()),
    ("AppleSilicon", "Apple Silicon GPU",   lambda: AppleSiliconPlugin()),
    ("NvidiaGPU",    "NVIDIA GPU",          lambda: NvidiaGPUPlugin()),
    ("AMDGPU",       "AMD GPU",             lambda: AMDGPUPlugin()),
    ("IntelGPU",     "Intel GPU",           lambda: IntelGPUPlugin()),
    ("TPU",          "Google TPU",          lambda: TPUPlugin()),
]


def discover_local() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for tag, pretty, factory in _DISCOVERABLE:
        try:
            plugin = factory()
            fp = plugin.probe()
        except Exception as exc:
            results.append({
                "tag": tag, "pretty": pretty, "available": False,
                "name": f"{pretty} (probe failed)", "error": str(exc)[:140],
                "plugin": None, "fp": None, "kind": "local",
                "max_qubits": 0, "compute_class": "?",
            })
            continue
        if fp is None:
            continue
        results.append({
            "tag": tag, "pretty": pretty, "available": bool(fp.is_available),
            "name": fp.name, "max_qubits": fp.max_qubits,
            "compute_class": getattr(fp.compute_class, "name", str(fp.compute_class)),
            "is_local": bool(fp.is_local),
            "plugin": plugin, "fp": fp, "kind": "local", "error": "",
        })
    results.append({
        "tag": "IonQ", "pretty": "IonQ Cloud QPU", "available": False,
        "name": "QPU::ionq  (remote)", "max_qubits": 36,
        "compute_class": "QUANTUM_HARDWARE", "is_local": False,
        "plugin": None, "fp": None, "kind": "cloud",
        "error": "link required — open Link Manager",
    })
    return results


TEMPLATES: List[Dict[str, str]] = [
    {
        "title": "GHZ-5 · CPU",
        "blurb": "5-qubit Greenberger–Horne–Zeilinger state on local statevector sim.",
        "code": (
            'n_qubits = 5\n'
            'seq = [("H",[0])] + [("CNOT",[i,i+1]) for i in range(4)]\n'
            'counts = CPUPlugin().execute_sample(n_qubits, seq, 2048)\n'
            'note = "GHZ-5 · local statevector"\n'
        ),
    },
    {
        "title": "AEGIS · redundant GHZ-5",
        "blurb": "AEGIS-Ion-N(12,7,3,1) collapsing 15 gates down to 5.",
        "code": (
            'src = "program dirty\\n" + "\\n".join(f"alloc q{i}" for i in range(5)) + "\\n"\n'
            'src += "gate H q0\\ngate H q0\\ngate H q0\\n"\n'
            'for i in range(1,5):\n'
            '    src += f"gate CX q{i-1} q{i}\\ngate CX q{i-1} q{i}\\ngate CX q{i-1} q{i}\\n"\n'
            'bridged = compile_qlambda_for_hardware(src)\n'
            'baseline = list(bridged.gate_sequence)\n'
            'n_qubits = bridged.n_qubits\n'
            'res = aegis_ion_nested(baseline, n_qubits)\n'
            'seq = res.winner.sequence\n'
            'counts = CPUPlugin().execute_sample(n_qubits, seq, 2048)\n'
            'note = f"{res.winner.strategy} · 2Q: {count_2q(baseline)}→{count_2q(seq)}"\n'
        ),
    },
    {
        "title": "IonQ · GHZ-3 (requires link)",
        "blurb": "Submit through the IonQ link forged in Link Manager.",
        "code": (
            'task = QuantumTask(n_qubits=3,\n'
            '    gate_sequence=[("H",[0]),("CNOT",[0,1]),("CNOT",[1,2])], shots=512)\n'
            'counts, link = links.submit(task, prefer=["ionq"])\n'
            'seq, n_qubits = list(task.gate_sequence), task.n_qubits\n'
            'note = f"GHZ-3 on {link.handle} · {link.health.latency_ms:.0f}ms · {link.target}"\n'
        ),
    },
    {
        "title": "IonQ Aria noise preview",
        "blurb": "Calibrated Pauli-noise Monte Carlo — no API spend.",
        "code": (
            'n_qubits = 7\n'
            'seq = [("H",[0])] + [("CNOT",[i,i+1]) for i in range(6)]\n'
            'counts = sample_with_ionq_noise(n_qubits, seq, 800, spec=IonQNoiseSpec.aria(), seed=1)\n'
            'note = f"GHZ-7 expected F={expected_circuit_fidelity(seq):.4f} @ Aria"\n'
        ),
    },
    {
        "title": "Hard TSP · AEGIS + IonQ (full run)",
        "blurb": "Runs the 4-city TSP driver — ≈30s, needs IonQ link.",
        "code": (
            'import runpy\n'
            'runpy.run_path("examples/aegis_ion_tsp_on_ionq.py", run_name="__main__")\n'
        ),
    },
    {
        "title": "Inspect session",
        "blurb": "Print loaded names and open links.",
        "code": (
            'print("links:", [l.handle for l in links.links()])\n'
            'print("locals:", sorted(k for k in dir() if not k.startswith("_"))[:32])\n'
        ),
    },
]


DOCS_LINKS = [
    ("IonQ Cloud console",   "https://cloud.ionq.com/"),
    ("IonQ API docs",        "https://docs.ionq.com/"),
    ("Qiskit-IonQ provider", "https://pypi.org/project/qiskit-ionq/"),
    ("Anthropic Claude API", "https://docs.anthropic.com/"),
]


# ═══════════════════════ helpers ══════════════════════════════════════════

def _fmt_ago(ts: float) -> str:
    if not ts:
        return "never"
    dt = time.time() - ts
    if dt < 1:
        return "just now"
    if dt < 60:
        return f"{int(dt)}s ago"
    if dt < 3600:
        return f"{int(dt/60)}m ago"
    return f"{int(dt/3600)}h ago"


def _state_color(state: str) -> str:
    return {
        "linked":      T["good"],
        "degraded":    T["warn"],
        "error":       T["err"],
        "handshaking": T["accent"],
        "unlinked":    T["fg_dim"],
        "closed":      T["fg_dim"],
    }.get(state, T["fg_dim"])


# ═══════════════════════ App window base ══════════════════════════════════

class AppWindow(ctk.CTkToplevel):
    """Base class for every vQPU app window.

    Registers with the taskbar, tracks its minimized/open state, and exposes
    a standard header.
    """

    APP_NAME: str = "App"
    APP_ICON: str = "■"
    APP_SIZE = (760, 520)

    def __init__(self, desktop: "Desktop") -> None:
        super().__init__(desktop)
        self.desktop = desktop
        self.title(f"vQPU — {self.APP_NAME}")
        w, h = self.APP_SIZE
        self.geometry(f"{w}x{h}")
        self.configure(fg_color=T["panel"])
        self.after(80, self._finalise)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.desktop.taskbar.register(self)

    def _finalise(self) -> None:
        try:
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    def _build(self) -> None:
        """Subclasses implement their UI here."""

    def _close(self) -> None:
        self.desktop.taskbar.unregister(self)
        self.destroy()

    def header(self, parent, subtitle: str = "") -> ctk.CTkFrame:
        h = ctk.CTkFrame(parent, fg_color=T["panel2"], corner_radius=0, height=54)
        h.pack(side="top", fill="x")
        h.pack_propagate(False)
        ctk.CTkLabel(h, text=f"{self.APP_ICON}  {self.APP_NAME}",
                     text_color=T["accent"], font=T["font_big"]
                     ).pack(side="left", padx=18)
        if subtitle:
            ctk.CTkLabel(h, text=subtitle, text_color=T["fg_dim"],
                         font=T["font_sm"]).pack(side="left", padx=6)
        return h


# ═══════════════════════ App: Terminal ════════════════════════════════════

class TerminalApp(AppWindow):
    APP_NAME = "Terminal"
    APP_ICON = "▶"
    APP_SIZE = (1020, 640)

    def _build(self) -> None:
        self.header(self, "persistent Python REPL · Enter to run · ⌘O open .py")

        body = ctk.CTkFrame(self, fg_color=T["panel"], corner_radius=0)
        body.pack(side="top", fill="both", expand=True, padx=14, pady=10)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.log = ctk.CTkTextbox(body, fg_color=T["bg"], text_color=T["fg"],
                                  font=T["font"], corner_radius=BTN_CORNER,
                                  border_width=0, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.tag_config("src",  foreground=T["accent"])
        self.log.tag_config("out",  foreground=T["fg"])
        self.log.tag_config("err",  foreground=T["err"])
        self.log.tag_config("meta", foreground=T["fg_dim"])
        self.log.tag_config("ok",   foreground=T["good"])
        self.log.configure(state="disabled")

        row = ctk.CTkFrame(self, fg_color=T["panel"])
        row.pack(side="top", fill="x", padx=14, pady=(4, 14))

        ctk.CTkLabel(row, text="Enter = run · Shift+Enter = newline · ⌘↵ also runs",
                     text_color=T["fg_dim"], font=T["font_sm"]).pack(anchor="w")

        inner = ctk.CTkFrame(row, fg_color=T["panel"])
        inner.pack(side="top", fill="x", pady=(4, 0))

        edit = ctk.CTkFrame(inner, fg_color=T["panel3"], corner_radius=BTN_CORNER,
                            border_width=1, border_color=T["border"])
        edit.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.editor = ctk.CTkTextbox(edit, fg_color=T["panel3"], text_color=T["fg"],
                                     font=T["font"], corner_radius=BTN_CORNER,
                                     border_width=0, wrap="word", height=110)
        self.editor.pack(fill="both", expand=True, padx=10, pady=8)
        self._install_bindings()

        col = ctk.CTkFrame(inner, fg_color=T["panel"])
        col.pack(side="left", fill="y")
        ctk.CTkButton(col, text="▶   RUN",
                      command=self._exec,
                      fg_color=T["accent"], hover_color=T["accent2"], text_color="#061018",
                      font=("SF Pro Display", 13, "bold"),
                      corner_radius=BTN_CORNER, height=42, width=120).pack(pady=(0, 6))
        ctk.CTkButton(col, text="Open .py",
                      command=self._open_file,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=120).pack(pady=3)
        ctk.CTkButton(col, text="Save log",
                      command=self._save_log,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=120).pack(pady=3)
        ctk.CTkButton(col, text="Clear",
                      command=self._clear,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=120).pack(pady=3)
        ctk.CTkButton(col, text="Reset session",
                      command=self._reset,
                      fg_color=T["panel3"], hover_color=T["err"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=120).pack(pady=3)

        self._pending: List[queue.Queue] = []

        # Allow external callers to load code into the editor.
        self.desktop.register_terminal(self)
        self._render_intro()

        self.bind("<Command-o>", lambda e: self._open_file())
        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Command-s>", lambda e: self._save_log())
        self.bind("<Control-s>", lambda e: self._save_log())
        self.bind("<Command-l>", lambda e: self._clear())
        self.bind("<Control-l>", lambda e: self._clear())
        self.after(150, lambda: self.editor.focus_set())

    def _install_bindings(self) -> None:
        def _run_break(_e=None): self._exec(); return "break"
        def _newline(_e=None): self.editor.insert("insert", "\n"); return "break"
        for seq in ("<Return>", "<KP_Enter>", "<Command-Return>", "<Control-Return>"):
            self.editor.bind(seq, _run_break)
        self.editor.bind("<Shift-Return>", _newline)

    def _render_intro(self) -> None:
        self._append("vQPU Terminal · persistent session\n", "meta")
        self._append("  vqpu, chesso, AEGIS, IonQNoiseSpec, links (LinkManager) are loaded.\n", "meta")
        self._append("  Tip: open Workloads to load a template with one click.\n", "meta")
        self._append("\n", "meta")

    def load_code(self, code_text: str) -> None:
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", code_text)
        self.editor.focus_set()
        self.lift()

    def _exec(self) -> None:
        source = self.editor.get("1.0", "end-1c")
        if not source.strip():
            return
        self.editor.delete("1.0", "end")
        self._append(">>> ", "src")
        self._append(source if source.endswith("\n") else source + "\n", "src")
        q = self.desktop.repl.submit(source)
        self._pending.append(q)
        self.after(30, self._poll)

    def _poll(self) -> None:
        still = []
        for q in self._pending:
            try:
                entry = q.get_nowait()
                self._render(entry)
            except queue.Empty:
                still.append(q)
        self._pending = still
        if still:
            self.after(80, self._poll)

    def _render(self, entry: Dict[str, Any]) -> None:
        if entry.get("stdout"):
            self._append(entry["stdout"], "out")
        if entry.get("stderr"):
            self._append(entry["stderr"], "err")
        if not entry.get("stdout") and not entry.get("stderr"):
            self._append("(no output)\n", "meta")
        self._append(f"  [{entry.get('elapsed_ms', 0.0):.0f} ms]\n\n", "meta")

    def _append(self, text: str, tag: str) -> None:
        self.log.configure(state="normal")
        try:
            self.log.insert("end", text, tag)
        except tk.TclError:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._render_intro()

    def _reset(self) -> None:
        self.desktop.repl.reset()
        self._append("--- REPL session reset ---\n", "meta")

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Python file",
            filetypes=[("Python", "*.py"), ("Text", "*.txt"), ("All", "*.*")],
            initialdir=str(pathlib.Path.cwd()),
        )
        if not path:
            return
        try:
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", pathlib.Path(path).read_text())
            self._append(f"[open] loaded {path}\n", "meta")
            self.editor.focus_set()
        except Exception as exc:
            self._append(f"[open] {exc}\n", "err")

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".txt",
            initialfile=f"vqpu_session_{int(time.time())}.txt",
        )
        if not path:
            return
        try:
            pathlib.Path(path).write_text(self.log.get("1.0", "end-1c"))
            self._append(f"[save] {path}\n", "meta")
        except Exception as exc:
            self._append(f"[save] {exc}\n", "err")

    def _close(self) -> None:
        self.desktop.register_terminal(None)
        super()._close()


# ═══════════════════════ App: Link Manager ════════════════════════════════

class LinkManagerApp(AppWindow):
    APP_NAME = "Link Manager"
    APP_ICON = "⚡"
    APP_SIZE = (720, 560)

    def _build(self) -> None:
        self.header(self, "quantum/classical link layer — like NVLink, for QPUs")

        top = ctk.CTkFrame(self, fg_color=T["panel"])
        top.pack(side="top", fill="x", padx=14, pady=(10, 0))
        ctk.CTkButton(top, text="⚡  Forge IonQ link",
                      command=self._forge,
                      fg_color=T["accent"], hover_color=T["accent2"], text_color="#061018",
                      font=T["font_hdr"], corner_radius=BTN_CORNER, height=40
                      ).pack(side="left")
        ctk.CTkButton(top, text="Re-heartbeat all",
                      command=self._heartbeat_all,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=40
                      ).pack(side="left", padx=8)

        self.list = ctk.CTkScrollableFrame(self, fg_color=T["panel"])
        self.list.pack(side="top", fill="both", expand=True, padx=14, pady=10)

        self._refresh()
        self._tick()

    def _forge(self) -> None:
        ForgeLinkDialog(self, self.desktop.manager, on_forged=lambda l: self._refresh())

    def _heartbeat_all(self) -> None:
        def _go():
            for l in list(self.desktop.manager.links()):
                try: l.heartbeat()
                except Exception: pass
            self.after(0, self._refresh)
        threading.Thread(target=_go, daemon=True).start()

    def _tick(self) -> None:
        self._refresh()
        self.after(4000, self._tick)

    def _refresh(self) -> None:
        for ch in self.list.winfo_children():
            ch.destroy()
        snaps = self.desktop.manager.snapshot()
        if not snaps:
            ctk.CTkLabel(self.list, text="No active links yet — click Forge IonQ link.",
                         text_color=T["fg_dim"], font=T["font_sm"]
                         ).pack(anchor="w", padx=8, pady=8)
            return
        for s in snaps:
            self._row(s)

    def _row(self, s: Dict[str, Any]) -> None:
        card = ctk.CTkFrame(self.list, fg_color=T["panel2"],
                            corner_radius=BTN_CORNER, border_width=1,
                            border_color=T["border"])
        card.pack(fill="x", padx=6, pady=6)
        head = ctk.CTkFrame(card, fg_color=T["panel2"])
        head.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(head, text="●", text_color=_state_color(s["state"]),
                     font=("SF Pro Display", 16)).pack(side="left")
        ctk.CTkLabel(head, text=f"  {s['handle']}  ", text_color=T["fg"],
                     font=T["font_hdr"]).pack(side="left")
        ctk.CTkLabel(head, text=f"[{s['kind']}]", text_color=T["fg_dim"],
                     font=T["font_xs"]).pack(side="left")
        ctk.CTkButton(head, text="Close", width=72, height=28,
                      command=lambda h=s["handle"]: self._close(h),
                      fg_color=T["panel3"], hover_color=T["err"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER).pack(side="right")
        ctk.CTkLabel(card, text=s["backend_name"], text_color=T["fg2"],
                     font=T["font_sm"], anchor="w").pack(fill="x", padx=14)
        stats = (f"state {s['state']} · lat {s['latency_ms']:.1f}ms · "
                 f"ok {s['calls_ok']} err {s['calls_err']} · "
                 f"beat {_fmt_ago(s['last_heartbeat_ts'])}")
        ctk.CTkLabel(card, text=stats, text_color=T["fg_dim"], font=T["font_xs"],
                     anchor="w").pack(fill="x", padx=14, pady=(0, 10))
        if s["last_error"]:
            ctk.CTkLabel(card, text=f"⚠  {s['last_error']}", text_color=T["warn"],
                         font=T["font_xs"], anchor="w", wraplength=560, justify="left"
                         ).pack(fill="x", padx=14, pady=(0, 10))

    def _close(self, handle: str) -> None:
        self.desktop.manager.close(handle)
        self._refresh()


# ═══════════════════════ App: Backends ════════════════════════════════════

class BackendsApp(AppWindow):
    APP_NAME = "Backends"
    APP_ICON = "◆"
    APP_SIZE = (720, 560)

    def _build(self) -> None:
        self.header(self, "discovered classical + quantum compute backends")
        top = ctk.CTkFrame(self, fg_color=T["panel"])
        top.pack(side="top", fill="x", padx=14, pady=(10, 0))
        ctk.CTkButton(top, text="Re-probe",
                      command=self._refresh,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=36
                      ).pack(side="left")

        self.list = ctk.CTkScrollableFrame(self, fg_color=T["panel"])
        self.list.pack(fill="both", expand=True, padx=14, pady=10)
        self._refresh()

    def _refresh(self) -> None:
        for ch in self.list.winfo_children():
            ch.destroy()
        for d in discover_local():
            self._card(d)

    def _card(self, d: Dict[str, Any]) -> None:
        linked = d["tag"].lower() in self.desktop.manager
        card = ctk.CTkFrame(self.list, fg_color=T["panel2"],
                            corner_radius=BTN_CORNER, border_width=1,
                            border_color=T["border"])
        card.pack(fill="x", padx=6, pady=6)
        top = ctk.CTkFrame(card, fg_color=T["panel2"])
        top.pack(fill="x", padx=12, pady=(10, 2))
        dot = T["good"] if linked else (T["accent"] if d["available"] else T["muted"])
        ctk.CTkLabel(top, text="●", text_color=dot, font=("SF Pro Display", 16)
                     ).pack(side="left")
        ctk.CTkLabel(top, text=f"  {d['pretty']}", text_color=T["fg"],
                     font=T["font_hdr"]).pack(side="left")
        ctk.CTkLabel(card, text=d["name"], text_color=T["fg2"], font=T["font_sm"],
                     anchor="w").pack(fill="x", padx=14)
        sub = (f"{d.get('compute_class','')} · {d.get('max_qubits','?')}q · "
               f"{'cloud' if d['kind']=='cloud' else 'local'}")
        ctk.CTkLabel(card, text=sub, text_color=T["fg_dim"], font=T["font_xs"],
                     anchor="w").pack(fill="x", padx=14, pady=(0, 2))
        if d["error"]:
            ctk.CTkLabel(card, text=d["error"], text_color=T["warn"], font=T["font_xs"],
                         anchor="w").pack(fill="x", padx=14, pady=(0, 4))
        if d["kind"] == "cloud" and not linked:
            ctk.CTkButton(card, text="⚡  Forge link",
                          command=lambda: self.desktop.open_app("link"),
                          fg_color=T["accent"], hover_color=T["accent2"],
                          text_color="#061018", font=T["font_ui"],
                          corner_radius=BTN_CORNER, height=32
                          ).pack(fill="x", padx=12, pady=(0, 10))
        else:
            ctk.CTkLabel(card, text="● linked" if linked else "ready",
                         text_color=(T["good"] if linked else T["fg_dim"]),
                         font=T["font_xs"], anchor="w"
                         ).pack(fill="x", padx=14, pady=(0, 10))


# ═══════════════════════ App: Workloads ═══════════════════════════════════

class WorkloadsApp(AppWindow):
    APP_NAME = "Workloads"
    APP_ICON = "☰"
    APP_SIZE = (720, 620)

    def _build(self) -> None:
        self.header(self, "one-click templates · loads into Terminal")

        scroll = ctk.CTkScrollableFrame(self, fg_color=T["panel"])
        scroll.pack(fill="both", expand=True, padx=14, pady=12)
        for tpl in TEMPLATES:
            self._card(scroll, tpl)

    def _card(self, parent, tpl: Dict[str, str]) -> None:
        card = ctk.CTkFrame(parent, fg_color=T["panel2"],
                            corner_radius=BTN_CORNER, border_width=1,
                            border_color=T["border"])
        card.pack(fill="x", padx=6, pady=6)
        row = ctk.CTkFrame(card, fg_color=T["panel2"])
        row.pack(fill="x", padx=14, pady=12)
        info = ctk.CTkFrame(row, fg_color=T["panel2"])
        info.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(info, text=tpl["title"], text_color=T["fg"], font=T["font_hdr"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=tpl["blurb"], text_color=T["fg_dim"], font=T["font_sm"],
                     anchor="w", wraplength=460, justify="left"
                     ).pack(anchor="w", pady=(2, 0))
        ctk.CTkButton(row, text="Load →",
                      command=lambda c=tpl["code"]: self._load(c),
                      fg_color=T["accent"], hover_color=T["accent2"], text_color="#061018",
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=96
                      ).pack(side="right", padx=6)
        ctk.CTkButton(row, text="Load & Run",
                      command=lambda c=tpl["code"]: self._load_run(c),
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=110
                      ).pack(side="right", padx=4)

    def _load(self, code_text: str) -> None:
        term: TerminalApp = self.desktop.open_app("terminal")
        self.after(50, lambda: term.load_code(code_text))

    def _load_run(self, code_text: str) -> None:
        term: TerminalApp = self.desktop.open_app("terminal")
        def _go():
            term.load_code(code_text)
            self.after(50, term._exec)
        self.after(80, _go)


# ═══════════════════════ App: Job Monitor ═════════════════════════════════

class JobMonitorApp(AppWindow):
    APP_NAME = "Job Monitor"
    APP_ICON = "▤"
    APP_SIZE = (820, 540)

    def _build(self) -> None:
        self.header(self, "every REPL execution — OK/err, elapsed, artifact peek")

        self.list = ctk.CTkScrollableFrame(self, fg_color=T["panel"])
        self.list.pack(fill="both", expand=True, padx=14, pady=12)

        self.desktop.repl.subscribe(self._on_job)
        self._refresh()

    def _on_job(self, entry: Dict[str, Any]) -> None:
        try:
            self.after(0, self._refresh)
        except tk.TclError:
            pass  # window closed

    def _refresh(self) -> None:
        for ch in self.list.winfo_children():
            ch.destroy()
        jobs = list(self.desktop.repl.job_log)[-40:][::-1]
        if not jobs:
            ctk.CTkLabel(self.list, text="No jobs yet — run something in Terminal.",
                         text_color=T["fg_dim"], font=T["font_sm"]
                         ).pack(anchor="w", padx=8, pady=8)
            return
        for entry in jobs:
            self._row(entry)

    def _row(self, entry: Dict[str, Any]) -> None:
        card = ctk.CTkFrame(self.list, fg_color=T["panel2"], corner_radius=BTN_CORNER,
                            border_width=1, border_color=T["border"])
        card.pack(fill="x", padx=6, pady=4)
        head = ctk.CTkFrame(card, fg_color=T["panel2"])
        head.pack(fill="x", padx=12, pady=(8, 2))
        color = T["good"] if entry["ok"] else T["err"]
        ctk.CTkLabel(head, text="●", text_color=color, font=("SF Pro Display", 14)
                     ).pack(side="left")
        when = time.strftime("%H:%M:%S", time.localtime(entry.get("t", 0)))
        ctk.CTkLabel(head, text=f"  {when}", text_color=T["fg_dim"],
                     font=T["font_sm"]).pack(side="left")
        ctk.CTkLabel(head, text=f"  {entry.get('elapsed_ms', 0):.0f} ms",
                     text_color=T["fg_dim"], font=T["font_xs"]).pack(side="right")
        src = entry["source"].strip().split("\n")[0][:100]
        ctk.CTkLabel(card, text=src, text_color=T["fg"], font=T["font_sm"],
                     anchor="w", wraplength=760, justify="left"
                     ).pack(fill="x", padx=14, pady=(0, 4))
        art = entry.get("artifacts") or {}
        meta = []
        if art.get("counts"):
            meta.append(f"counts({len(art['counts'])})")
        if art.get("seq"):
            s = art["seq"]
            meta.append(f"seq len={s['length']} 2Q={s['two_q']}")
        if art.get("n_qubits"):
            meta.append(f"{art['n_qubits']}q")
        if entry.get("stderr"):
            meta.append("⚠ stderr")
        if meta:
            ctk.CTkLabel(card, text=" · ".join(meta), text_color=T["accent2"],
                         font=T["font_xs"], anchor="w"
                         ).pack(fill="x", padx=14, pady=(0, 8))


# ═══════════════════════ App: Circuit Viewer ══════════════════════════════

class CircuitViewerApp(AppWindow):
    APP_NAME = "Circuit Viewer"
    APP_ICON = "⎓"
    APP_SIZE = (820, 520)

    def _build(self) -> None:
        self.header(self, "last `seq` from the REPL — grouped by qubit wire")
        self.body = ctk.CTkTextbox(self, fg_color=T["bg"], text_color=T["fg"],
                                   font=T["font"], corner_radius=BTN_CORNER,
                                   border_width=0, wrap="none")
        self.body.pack(fill="both", expand=True, padx=14, pady=12)
        ctk.CTkButton(self, text="Refresh",
                      command=self._refresh,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34
                      ).pack(side="bottom", pady=(0, 10))
        self._refresh()

    def _refresh(self) -> None:
        ns = self.desktop.repl.interp.locals
        seq = ns.get("seq")
        n = ns.get("n_qubits")
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        if not isinstance(seq, list) or not seq:
            self.body.insert("end", "No `seq` in the session.\nAssign a gate list to `seq`.\n")
            self.body.configure(state="disabled")
            return
        if not isinstance(n, int) or n <= 0:
            n = 1 + max((t for g in seq for t in (g[1] if isinstance(g[1], (list, tuple)) else [g[1]])), default=0)
        lanes = [[] for _ in range(n)]
        for i, g in enumerate(seq):
            name = str(g[0])
            raw_t = g[1]
            targets = list(raw_t) if isinstance(raw_t, (list, tuple)) else [raw_t]
            for q in range(n):
                if q in targets:
                    if len(targets) == 1:
                        lanes[q].append(f"[{name}]")
                    else:
                        role = "●" if q == targets[0] else "⊕"
                        lanes[q].append(f"[{name}{role}]")
                else:
                    lanes[q].append("──────")
        self.body.insert("end", f"n_qubits={n}  ops={len(seq)}  2Q={count_2q(seq)}\n\n")
        for q, lane in enumerate(lanes):
            self.body.insert("end", f"q{q:>2d}: " + " ".join(lane) + "\n")
        self.body.configure(state="disabled")


# ═══════════════════════ App: Docs ════════════════════════════════════════

class DocsApp(AppWindow):
    APP_NAME = "Docs"
    APP_ICON = "⌘"
    APP_SIZE = (620, 420)

    def _build(self) -> None:
        self.header(self, "useful web links — opens in your system browser")
        body = ctk.CTkScrollableFrame(self, fg_color=T["panel"])
        body.pack(fill="both", expand=True, padx=14, pady=12)
        for label, url in DOCS_LINKS:
            self._row(body, label, url)

    def _row(self, parent, label: str, url: str) -> None:
        card = ctk.CTkFrame(parent, fg_color=T["panel2"], corner_radius=BTN_CORNER,
                            border_width=1, border_color=T["border"])
        card.pack(fill="x", padx=6, pady=5)
        row = ctk.CTkFrame(card, fg_color=T["panel2"])
        row.pack(fill="x", padx=14, pady=10)
        info = ctk.CTkFrame(row, fg_color=T["panel2"])
        info.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(info, text=label, text_color=T["fg"], font=T["font_hdr"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=url, text_color=T["fg_dim"], font=T["font_xs"],
                     anchor="w").pack(anchor="w", pady=(2, 0))
        ctk.CTkButton(row, text="Open ↗",
                      command=lambda u=url: webbrowser.open(u),
                      fg_color=T["accent"], hover_color=T["accent2"], text_color="#061018",
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=34, width=90,
                      ).pack(side="right")


# ═══════════════════════ Forge Link dialog ════════════════════════════════

class ForgeLinkDialog(ctk.CTkToplevel):
    def __init__(self, master, manager: LinkManager, on_forged) -> None:
        super().__init__(master)
        self.manager = manager
        self.on_forged = on_forged
        self.title("Forge IonQ Link")
        self.geometry("640x460")
        self.resizable(False, False)
        self.configure(fg_color=T["panel"])
        self.transient(master)
        self.after(80, self.grab_set)
        self._busy = False

        hdr = ctk.CTkFrame(self, fg_color=T["panel2"], corner_radius=0, height=62)
        hdr.pack(side="top", fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="⚡  Forge IonQ Link", text_color=T["accent"],
                     font=T["font_big"]).pack(side="left", padx=20)
        ctk.CTkLabel(hdr, text="Enter to connect  ·  Esc to cancel",
                     text_color=T["fg_dim"], font=T["font_sm"]).pack(side="left", padx=6)

        body = ctk.CTkFrame(self, fg_color=T["panel"])
        body.pack(fill="both", expand=True, padx=22, pady=14)

        def _lbl(r, t):
            ctk.CTkLabel(body, text=t, text_color=T["fg2"], font=T["font_ui"],
                         anchor="w").grid(row=r, column=0, sticky="w", pady=8)
        def _ent(r, v, show=""):
            e = ctk.CTkEntry(body, textvariable=v, show=show, fg_color=T["panel3"],
                             border_color=T["border"], text_color=T["fg"],
                             placeholder_text_color=T["fg_dim"], corner_radius=BTN_CORNER,
                             font=T["font"], height=40)
            e.grid(row=r, column=1, sticky="we", padx=(14, 0), pady=8)
            return e
        def _combo(r, v, vs):
            c = ctk.CTkComboBox(body, variable=v, values=vs, fg_color=T["panel3"],
                                border_color=T["border"], button_color=T["panel3"],
                                button_hover_color=T["border"],
                                dropdown_fg_color=T["panel3"], dropdown_text_color=T["fg"],
                                text_color=T["fg"], corner_radius=BTN_CORNER,
                                font=T["font"], height=40)
            c.grid(row=r, column=1, sticky="we", padx=(14, 0), pady=8)
            return c

        _lbl(0, "API key")
        self.key_var = tk.StringVar(value=os.environ.get("IONQ_API_KEY", ""))
        self.key_entry = _ent(0, self.key_var, show="•")
        self.key_entry.configure(placeholder_text="paste ionq API key")

        _lbl(1, "Target backend")
        self.target_var = tk.StringVar(value=os.environ.get("IONQ_BACKEND", "ionq_simulator"))
        _combo(1, self.target_var, ["ionq_simulator", "qpu.aria-1",
                                     "qpu.forte-1", "qpu.forte-enterprise-1"])

        _lbl(2, "Noise model")
        self.noise_var = tk.StringVar(value=os.environ.get("IONQ_NOISE_MODEL", "aria-1"))
        _combo(2, self.noise_var, ["", "aria-1", "forte-1", "harmony"])

        _lbl(3, "Handle")
        self.handle_var = tk.StringVar(value="ionq")
        _ent(3, self.handle_var)

        body.columnconfigure(1, weight=1)

        self.status_lbl = ctk.CTkLabel(body, text="Paste the key and hit Enter.",
                                       text_color=T["fg_dim"], font=T["font_ui"],
                                       wraplength=560, justify="left", anchor="w")
        self.status_lbl.grid(row=4, column=0, columnspan=2, sticky="we", pady=(14, 2))

        act = ctk.CTkFrame(self, fg_color=T["panel2"], corner_radius=0, height=76)
        act.pack(side="bottom", fill="x")
        act.pack_propagate(False)
        self.connect = ctk.CTkButton(act, text="⚡   CONNECT   (Enter)",
                                     command=self._go,
                                     fg_color=T["accent"], hover_color=T["accent2"],
                                     text_color="#061018",
                                     font=("SF Pro Display", 14, "bold"),
                                     corner_radius=BTN_CORNER, height=44, width=280)
        self.connect.pack(side="right", padx=16, pady=14)
        ctk.CTkButton(act, text="Cancel", command=self.destroy,
                      fg_color=T["panel3"], hover_color=T["border"], text_color=T["fg"],
                      font=T["font_ui"], corner_radius=BTN_CORNER, height=44, width=110
                      ).pack(side="right", padx=4, pady=14)

        for s in ("<Return>", "<KP_Enter>"):
            self.bind(s, lambda e: self._go())
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(100, lambda: (self.key_entry.focus_set(), self.key_entry.icursor("end")))

    def _set(self, msg, kind="dim"):
        color = {"dim": T["fg_dim"], "err": T["err"],
                 "ok": T["good"], "busy": T["accent"]}[kind]
        self.status_lbl.configure(text=msg, text_color=color)
        self.update_idletasks()

    def _go(self) -> None:
        if self._busy:
            return
        key = self.key_var.get().strip()
        target = self.target_var.get().strip()
        noise = (self.noise_var.get() or "").strip() or None
        handle = self.handle_var.get().strip() or "ionq"
        if not key:
            self._set("API key is required — paste it above, hit Enter.", "err")
            self.key_entry.focus_set()
            return
        self._busy = True
        self.connect.configure(state="disabled", text="…  CONNECTING")
        self._set(f"handshaking with {target} …", "busy")

        def _work():
            try:
                link = self.manager.forge_ionq(
                    handle=handle, api_key=key, target_backend=target, noise_model=noise,
                )
                err = None if link.state == LinkState.LINKED else (
                    f"link went to {link.state.value}: {link.health.last_error[:200]}"
                )
            except Exception as exc:
                link, err = None, f"forge failed: {exc}"
            self.after(0, lambda: self._done(link, err))

        threading.Thread(target=_work, daemon=True).start()

    def _done(self, link, err) -> None:
        self._busy = False
        self.connect.configure(state="normal", text="⚡   CONNECT   (Enter)")
        if err is not None or link is None:
            self._set(err or "unknown forge error", "err")
            return
        self._set(f"linked → {link.target}  ({link.health.latency_ms:.0f} ms)", "ok")
        self.on_forged(link)
        self.after(500, self.destroy)


# ═══════════════════════ Taskbar ══════════════════════════════════════════

class Taskbar(ctk.CTkFrame):
    def __init__(self, desktop: "Desktop") -> None:
        super().__init__(desktop, fg_color=T["taskbar"], corner_radius=0, height=46,
                         border_width=0)
        self.desktop = desktop
        self.pack_propagate(False)
        self._apps: Dict[AppWindow, ctk.CTkButton] = {}
        self._left = ctk.CTkFrame(self, fg_color=T["taskbar"])
        self._left.pack(side="left", fill="y", padx=10)
        self._right = ctk.CTkFrame(self, fg_color=T["taskbar"])
        self._right.pack(side="right", fill="y", padx=14)
        self._tray = ctk.CTkLabel(self._right, text="", text_color=T["fg_dim"],
                                  font=T["font_sm"])
        self._tray.pack(side="right", padx=6)
        self._clock = ctk.CTkLabel(self._right, text="", text_color=T["fg_dim"],
                                   font=T["font_sm"])
        self._clock.pack(side="right", padx=6)
        self._tick_clock()

    def register(self, app: AppWindow) -> None:
        btn = ctk.CTkButton(self._left, text=f"{app.APP_ICON}  {app.APP_NAME}",
                            command=lambda a=app: self._raise(a),
                            fg_color=T["panel3"], hover_color=T["border"],
                            text_color=T["fg"], font=T["font_ui"],
                            corner_radius=BTN_CORNER, height=32, width=140)
        btn.pack(side="left", padx=3, pady=6)
        self._apps[app] = btn

    def unregister(self, app: AppWindow) -> None:
        btn = self._apps.pop(app, None)
        if btn is not None:
            btn.destroy()

    def _raise(self, app: AppWindow) -> None:
        try:
            app.deiconify()
            app.lift()
            app.focus_force()
        except tk.TclError:
            self.unregister(app)

    def set_tray(self, text: str) -> None:
        self._tray.configure(text=text)

    def _tick_clock(self) -> None:
        self._clock.configure(text=time.strftime("%a %H:%M:%S"))
        self.after(1000, self._tick_clock)


# ═══════════════════════ Desktop shell ════════════════════════════════════

APPS: List[Dict[str, Any]] = [
    {"key": "terminal",  "name": "Terminal",       "icon": "▶",  "cls": TerminalApp,
     "blurb": "persistent Python REPL"},
    {"key": "link",      "name": "Link Manager",   "icon": "⚡", "cls": LinkManagerApp,
     "blurb": "forge & monitor quantum links"},
    {"key": "backends",  "name": "Backends",       "icon": "◆",  "cls": BackendsApp,
     "blurb": "discover CPU · GPU · QPU"},
    {"key": "workloads", "name": "Workloads",      "icon": "☰",  "cls": WorkloadsApp,
     "blurb": "one-click templates"},
    {"key": "monitor",   "name": "Job Monitor",    "icon": "▤",  "cls": JobMonitorApp,
     "blurb": "live REPL execution log"},
    {"key": "circuit",   "name": "Circuit Viewer", "icon": "⎓",  "cls": CircuitViewerApp,
     "blurb": "visualize current `seq`"},
    {"key": "docs",      "name": "Docs",           "icon": "⌘",  "cls": DocsApp,
     "blurb": "open web docs / consoles"},
]


class Desktop(ctk.CTk):
    HEARTBEAT_S = 5.0

    def __init__(self) -> None:
        super().__init__()
        self.title("vQPU")
        self.geometry("1440x900")
        self.minsize(1100, 680)
        self.configure(fg_color=T["bg"])

        self.manager = LinkManager()
        self.repl = ReplEngine(self.manager)
        self._open: Dict[str, AppWindow] = {}
        self._terminal: Optional[TerminalApp] = None

        self._build_top_bar()
        self._build_desktop()
        self.taskbar = Taskbar(self)
        self.taskbar.pack(side="bottom", fill="x")

        self._autolink_locals()
        self._update_tray()
        self._schedule_heartbeat()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Open Terminal by default so the user has somewhere to run code.
        self.after(300, lambda: self.open_app("terminal"))

    # --- top bar ------------------------------------------------------
    def _build_top_bar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=T["panel"], corner_radius=0, height=52)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="vQPU", text_color=T["accent"],
                     font=("SF Pro Display", 22, "bold")).pack(side="left", padx=18)
        ctk.CTkLabel(bar, text="   quantum virtual machine  ·  VM Desktop",
                     text_color=T["fg_dim"], font=T["font_ui"]).pack(side="left")

        right = ctk.CTkFrame(bar, fg_color=T["panel"])
        right.pack(side="right", padx=14)
        for app in APPS:
            ctk.CTkButton(
                right, text=f"{app['icon']}", width=36, height=32,
                command=lambda k=app["key"]: self.open_app(k),
                fg_color=T["panel3"], hover_color=T["accent"],
                text_color=T["fg"], font=("SF Pro Display", 14, "bold"),
                corner_radius=BTN_CORNER,
            ).pack(side="left", padx=2)

    # --- desktop wallpaper with app icons ----------------------------
    def _build_desktop(self) -> None:
        wrap = ctk.CTkFrame(self, fg_color=T["bg"], corner_radius=0)
        wrap.pack(fill="both", expand=True)
        # Gentle "wallpaper" frame with a subtle border to feel like a workspace
        canvas = tk.Canvas(wrap, bg=T["bg"], highlightthickness=0, borderwidth=0)
        canvas.pack(fill="both", expand=True)

        grid_host = ctk.CTkFrame(canvas, fg_color=T["bg"])
        grid_host.place(relx=0.5, rely=0.5, anchor="center")

        # Banner title
        ctk.CTkLabel(grid_host, text="quantum · classical · in one shell",
                     text_color=T["fg_dim"], font=("SF Pro Display", 14)
                     ).grid(row=0, column=0, columnspan=4, pady=(0, 26))

        per_row = 4
        for i, app in enumerate(APPS):
            r = 1 + i // per_row
            c = i % per_row
            self._icon_tile(grid_host, app).grid(row=r, column=c, padx=22, pady=22)

    def _icon_tile(self, parent, app: Dict[str, Any]) -> ctk.CTkFrame:
        tile = ctk.CTkFrame(parent, fg_color=T["panel"], corner_radius=18,
                            border_width=1, border_color=T["border"],
                            width=170, height=150)
        tile.grid_propagate(False)
        tile.columnconfigure(0, weight=1)
        ctk.CTkLabel(tile, text=app["icon"], text_color=T["accent"],
                     font=("SF Pro Display", 42, "bold")
                     ).grid(row=0, column=0, pady=(18, 4))
        ctk.CTkLabel(tile, text=app["name"], text_color=T["fg"], font=T["font_hdr"]
                     ).grid(row=1, column=0, pady=(2, 2))
        ctk.CTkLabel(tile, text=app["blurb"], text_color=T["fg_dim"],
                     font=("SF Pro Text", 10), wraplength=150, justify="center"
                     ).grid(row=2, column=0, pady=(0, 6))

        def _click(_e=None):
            self.open_app(app["key"])

        for w in (tile,) + tuple(tile.winfo_children()):
            w.bind("<Button-1>", _click)
            try:
                w.configure(cursor="hand2")
            except tk.TclError:
                pass
        return tile

    # --- app lifecycle ------------------------------------------------
    def open_app(self, key: str) -> Optional[AppWindow]:
        existing = self._open.get(key)
        if existing is not None:
            try:
                existing.lift()
                existing.focus_force()
                return existing
            except tk.TclError:
                self._open.pop(key, None)
        for app in APPS:
            if app["key"] != key:
                continue
            win = app["cls"](self)
            self._open[key] = win
            def _forget(_e=None, k=key):
                self._open.pop(k, None)
            win.bind("<Destroy>", _forget)
            return win
        return None

    def register_terminal(self, term: Optional[TerminalApp]) -> None:
        self._terminal = term

    # --- system tray --------------------------------------------------
    def _update_tray(self) -> None:
        snaps = self.manager.snapshot()
        linked = sum(1 for s in snaps if s["state"] == "linked")
        total = len(snaps)
        cloud = any(s["kind"] == "cloud" and s["state"] == "linked" for s in snaps)
        parts = [f"{linked}/{total} linked"]
        if cloud:
            parts.append("⚡ cloud up")
        self.taskbar.set_tray("  ·  ".join(parts))

    # --- heartbeat loop -----------------------------------------------
    def _schedule_heartbeat(self) -> None:
        self.after(int(self.HEARTBEAT_S * 1000), self._heartbeat)

    def _heartbeat(self) -> None:
        def _beat():
            for link in list(self.manager.links()):
                try:
                    link.heartbeat()
                except Exception:
                    pass
            self.after(0, self._update_tray)
        threading.Thread(target=_beat, daemon=True).start()
        self._schedule_heartbeat()

    # --- discovery --------------------------------------------------
    def _autolink_locals(self) -> None:
        for d in discover_local():
            if d["kind"] != "local" or not d["available"] or d["plugin"] is None:
                continue
            handle = d["tag"].lower()
            try:
                self.manager.forge_local(handle, d["plugin"], provider=d["tag"])
            except Exception:
                pass

    # --- shutdown ----------------------------------------------------
    def _on_close(self) -> None:
        try:
            self.manager.close_all()
        except Exception:
            pass
        self.destroy()


def main() -> int:
    app = Desktop()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
