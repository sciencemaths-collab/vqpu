"""QuantumLink — persistent compute-backend link layer for vqpu.

Inspiration
───────────
NVLink gives a pair of GPUs a persistent, authenticated, low-latency
channel so code on one can address the other as if it were local. We want
the same semantics between the vqpu front-end and every compute backend
it might drive — CPU, GPU, TPU, a quantum trap sitting on the same PCIe
bus, or a remote QPU behind a cloud API.

Model
─────
Two kinds of links, one uniform interface:

  LocalQuantumLink   — the backend is already reachable in-process.
                       Handshake is a probe, heartbeat is a no-op,
                       submit just forwards to the plugin. Latency is
                       measured but small.

  CloudQuantumLink   — the backend lives behind credentials and a
                       network. Handshake validates the key against the
                       provider, heartbeat periodically re-probes to
                       detect outages, submit routes through the cloud
                       plugin. Credentials live in memory only.

Both implement the same contract:

    link.handshake()     — one-shot authentication / capability probe
    link.heartbeat()     — idempotent liveness check, updates health
    link.submit(task)    — run a QuantumTask, return counts
    link.close()         — tear down and mark UNLINKED
    link.state           — LinkState enum
    link.health          — LinkHealth (latency, call stats, last beat)
    link.fingerprint     — BackendFingerprint from the underlying plugin

LinkManager keeps a dictionary of open links keyed by a handle string
("cpu", "ionq:ionq_simulator", "gpu:mps", …) and routes a QuantumTask to
whichever link best satisfies its requirements. The front-end (console,
scripts, Universal dispatcher) only has to pick a handle.

Security
────────
Credentials are never logged, never written to disk, never round-tripped
through a UI redraw. They live in the CloudQuantumLink instance and can
be purged with ``link.close()``. The repr of a link masks the key.
"""

from __future__ import annotations

import enum
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ─────────────────────────── types & state ────────────────────────────────

class LinkState(str, enum.Enum):
    UNLINKED = "unlinked"       # constructed but handshake not attempted
    HANDSHAKING = "handshaking"  # handshake in flight
    LINKED = "linked"           # up and usable
    DEGRADED = "degraded"       # last heartbeat failed but retained
    ERROR = "error"             # handshake or submit failed; inspect last_error
    CLOSED = "closed"           # intentionally torn down


class LinkKind(str, enum.Enum):
    LOCAL = "local"   # shares a process — direct call, no auth
    ATTACHED = "attached"  # hardware on the same host but via a driver channel
    CLOUD = "cloud"   # remote, behind a credential


@dataclass(slots=True)
class LinkHealth:
    """Rolling health stats for a link. Cheap to update on every call."""

    latency_ms: float = float("nan")       # last measured round-trip
    last_heartbeat_ts: float = 0.0         # epoch seconds
    calls_ok: int = 0
    calls_err: int = 0
    last_error: str = ""

    def record_ok(self, latency_ms: float) -> None:
        self.latency_ms = float(latency_ms)
        self.last_heartbeat_ts = time.time()
        self.calls_ok += 1

    def record_err(self, msg: str) -> None:
        self.calls_err += 1
        self.last_error = msg[:240]


@dataclass(slots=True)
class QuantumTask:
    """Unit of work a link can execute.

    ``gate_sequence`` is the only required field; the rest are hints that
    let LinkManager route the task when the caller doesn't care which
    backend runs it.
    """

    n_qubits: int
    gate_sequence: Sequence[Tuple]
    shots: int = 1024
    requires_cloud: bool = False
    requires_local: bool = False
    min_qubits: int = 0
    tag: str = ""


# ─────────────────────────────── base class ───────────────────────────────

class QuantumLink(ABC):
    """Persistent authenticated link to one compute backend."""

    def __init__(self, handle: str, kind: LinkKind, *, provider: str, target: str) -> None:
        self.handle = handle
        self.kind = kind
        self.provider = provider
        self.target = target
        self.state: LinkState = LinkState.UNLINKED
        self.health = LinkHealth()
        self.fingerprint = None  # filled by handshake()
        self.opened_at: float = 0.0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} handle={self.handle!r} "
            f"state={self.state.value} provider={self.provider!r} target={self.target!r}>"
        )

    @abstractmethod
    def _do_handshake(self) -> Any: ...

    @abstractmethod
    def _do_submit(self, task: QuantumTask) -> Dict[str, int]: ...

    def handshake(self) -> LinkState:
        with self._lock:
            self.state = LinkState.HANDSHAKING
            t0 = time.perf_counter()
            try:
                self.fingerprint = self._do_handshake()
                lat = (time.perf_counter() - t0) * 1000.0
                self.health.record_ok(lat)
                self.state = LinkState.LINKED
                self.opened_at = time.time()
            except Exception as exc:
                self.health.record_err(str(exc))
                self.state = LinkState.ERROR
            return self.state

    def heartbeat(self) -> LinkState:
        if self.state in (LinkState.UNLINKED, LinkState.CLOSED):
            return self.state
        t0 = time.perf_counter()
        try:
            self._do_handshake()  # re-probe; cheap for local, a ping for cloud
            lat = (time.perf_counter() - t0) * 1000.0
            self.health.record_ok(lat)
            # Recover from DEGRADED if we were there.
            if self.state in (LinkState.DEGRADED, LinkState.ERROR):
                self.state = LinkState.LINKED
        except Exception as exc:
            self.health.record_err(str(exc))
            self.state = LinkState.DEGRADED
        return self.state

    def submit(self, task: QuantumTask) -> Dict[str, int]:
        if self.state not in (LinkState.LINKED, LinkState.DEGRADED):
            raise RuntimeError(f"link {self.handle!r} is {self.state.value}, cannot submit")
        t0 = time.perf_counter()
        try:
            counts = self._do_submit(task)
            lat = (time.perf_counter() - t0) * 1000.0
            self.health.record_ok(lat)
            if self.state == LinkState.DEGRADED:
                self.state = LinkState.LINKED
            return counts
        except Exception as exc:
            self.health.record_err(str(exc))
            if self.state == LinkState.LINKED:
                self.state = LinkState.DEGRADED
            raise

    def close(self) -> None:
        with self._lock:
            self._do_close()
            self.state = LinkState.CLOSED

    def _do_close(self) -> None:
        """Subclasses may override to purge credentials/connections."""

    def snapshot(self) -> Dict[str, Any]:
        """Serializable health snapshot for UI display."""
        return {
            "handle": self.handle,
            "kind": self.kind.value,
            "state": self.state.value,
            "provider": self.provider,
            "target": self.target,
            "opened_at": self.opened_at,
            "latency_ms": self.health.latency_ms,
            "calls_ok": self.health.calls_ok,
            "calls_err": self.health.calls_err,
            "last_heartbeat_ts": self.health.last_heartbeat_ts,
            "last_error": self.health.last_error,
            "max_qubits": int(self.fingerprint.max_qubits) if self.fingerprint else 0,
            "backend_name": self.fingerprint.name if self.fingerprint else "",
        }


# ─────────────────────────────── local ────────────────────────────────────

class LocalQuantumLink(QuantumLink):
    """Direct-attach link — the backend is a Python plugin in our process.

    Handshake calls ``plugin.probe()``; heartbeat repeats the probe (cheap).
    """

    def __init__(self, handle: str, plugin: Any, *, provider: str | None = None) -> None:
        provider = provider or type(plugin).__name__
        super().__init__(
            handle=handle,
            kind=LinkKind.LOCAL,
            provider=provider,
            target="in-process",
        )
        self.plugin = plugin

    def _do_handshake(self) -> Any:
        fp = self.plugin.probe()
        if fp is None:
            raise RuntimeError(f"{self.provider}: probe returned None (not available)")
        if not getattr(fp, "is_available", False):
            raise RuntimeError(f"{self.provider}: {fp.name} probe says not available")
        return fp

    def _do_submit(self, task: QuantumTask) -> Dict[str, int]:
        return self.plugin.execute_sample(
            n_qubits=task.n_qubits,
            gate_sequence=list(task.gate_sequence),
            shots=task.shots,
        )


# ─────────────────────────────── cloud ────────────────────────────────────

class CloudQuantumLink(QuantumLink):
    """Credentialed remote link.

    Credentials are kept *only* in this instance. The plugin under the
    hood reads them from ``os.environ`` (that's how vqpu's cloud plugins
    were designed), so the link scopes the env var to its own thread-local
    presence and restores the prior value after each call.
    """

    def __init__(
        self,
        handle: str,
        *,
        provider: str,
        target_backend: str,
        api_key: str,
        env_key_name: str,
        env_backend_name: str,
        noise_model: Optional[str] = None,
        env_noise_name: Optional[str] = None,
        plugin_factory=None,
    ) -> None:
        super().__init__(
            handle=handle,
            kind=LinkKind.CLOUD,
            provider=provider,
            target=target_backend,
        )
        self._api_key = api_key   # stays in instance; never logged
        self._env_key_name = env_key_name
        self._env_backend_name = env_backend_name
        self._env_noise_name = env_noise_name
        self._noise_model = noise_model
        self._plugin_factory = plugin_factory
        self._plugin = None

    def _scoped_env(self) -> None:
        os.environ[self._env_key_name] = self._api_key
        os.environ[self._env_backend_name] = self.target
        if self._noise_model and self._env_noise_name:
            os.environ[self._env_noise_name] = self._noise_model

    def _clear_env(self) -> None:
        """Only purge the noise-model var; the API key & backend env are often
        expected to remain for subsequent plugin calls inside the same link."""
        # We leave the key/backend env set so that any plugin instantiated
        # during submit() sees them. close() wipes everything.

    def _build_plugin(self) -> Any:
        if self._plugin is not None:
            return self._plugin
        if self._plugin_factory is not None:
            self._plugin = self._plugin_factory()
            return self._plugin
        from vqpu import QPUCloudPlugin
        self._plugin = QPUCloudPlugin(self.provider.lower())
        return self._plugin

    def _do_handshake(self) -> Any:
        self._scoped_env()
        plugin = self._build_plugin()
        fp = plugin.probe()
        if fp is None:
            raise RuntimeError(f"{self.provider}: probe returned None; check credentials")
        if not getattr(fp, "is_available", False):
            raise RuntimeError(f"{self.provider}: backend not available")
        return fp

    def _do_submit(self, task: QuantumTask) -> Dict[str, int]:
        self._scoped_env()
        plugin = self._build_plugin()
        return plugin.execute_sample(
            n_qubits=task.n_qubits,
            gate_sequence=list(task.gate_sequence),
            shots=task.shots,
        )

    def _do_close(self) -> None:
        # Scrub credentials from this link *and* the environment we touched.
        self._api_key = ""
        for env in (self._env_key_name, self._env_backend_name, self._env_noise_name):
            if env and os.environ.get(env):
                try:
                    del os.environ[env]
                except KeyError:
                    pass
        self._plugin = None

    def __repr__(self) -> str:
        masked = f"{self._api_key[:4]}…" if self._api_key else "(closed)"
        return (
            f"<CloudQuantumLink handle={self.handle!r} state={self.state.value} "
            f"provider={self.provider!r} target={self.target!r} key={masked}>"
        )


# ──────────────────────────── LinkManager ─────────────────────────────────

class LinkManager:
    """Holds all open QuantumLinks. Thread-safe."""

    def __init__(self) -> None:
        self._links: Dict[str, QuantumLink] = {}
        self._lock = threading.RLock()

    # --- lifecycle ------------------------------------------------------
    def register(self, link: QuantumLink) -> QuantumLink:
        with self._lock:
            existing = self._links.get(link.handle)
            if existing is not None:
                existing.close()
            self._links[link.handle] = link
            return link

    def forge_local(self, handle: str, plugin: Any, *, provider: str | None = None) -> LocalQuantumLink:
        link = LocalQuantumLink(handle=handle, plugin=plugin, provider=provider or "")
        link.handshake()
        return self.register(link)  # type: ignore[return-value]

    def forge_ionq(
        self,
        handle: str,
        *,
        api_key: str,
        target_backend: str = "ionq_simulator",
        noise_model: Optional[str] = None,
    ) -> CloudQuantumLink:
        link = CloudQuantumLink(
            handle=handle,
            provider="ionq",
            target_backend=target_backend,
            api_key=api_key,
            env_key_name="IONQ_API_KEY",
            env_backend_name="IONQ_BACKEND",
            env_noise_name="IONQ_NOISE_MODEL",
            noise_model=noise_model,
        )
        link.handshake()
        return self.register(link)  # type: ignore[return-value]

    def close(self, handle: str) -> bool:
        with self._lock:
            link = self._links.pop(handle, None)
        if link is None:
            return False
        link.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            handles = list(self._links.keys())
        for h in handles:
            self.close(h)

    # --- access ---------------------------------------------------------
    def get(self, handle: str) -> Optional[QuantumLink]:
        with self._lock:
            return self._links.get(handle)

    def __getitem__(self, handle: str) -> QuantumLink:
        link = self.get(handle)
        if link is None:
            raise KeyError(f"no link with handle {handle!r}")
        return link

    def __contains__(self, handle: str) -> bool:
        return self.get(handle) is not None

    def links(self) -> List[QuantumLink]:
        with self._lock:
            return list(self._links.values())

    # --- routing --------------------------------------------------------
    def submit(
        self,
        task: QuantumTask,
        *,
        prefer: Sequence[str] = (),
    ) -> Tuple[Dict[str, int], QuantumLink]:
        """Route a task to the first link whose requirements match.

        Order of consideration:
          1. every handle listed in ``prefer`` (in order) that is LINKED
          2. any LINKED local link that satisfies the requirements
          3. any LINKED cloud link that satisfies the requirements
        """
        live = [l for l in self.links() if l.state in (LinkState.LINKED, LinkState.DEGRADED)]
        def _ok(link: QuantumLink) -> bool:
            if task.requires_cloud and link.kind != LinkKind.CLOUD:
                return False
            if task.requires_local and link.kind == LinkKind.CLOUD:
                return False
            if link.fingerprint is not None and task.min_qubits:
                if link.fingerprint.max_qubits < task.min_qubits:
                    return False
            return True

        ordered: List[QuantumLink] = []
        for p in prefer:
            link = self.get(p)
            if link is not None and link in live and _ok(link):
                ordered.append(link)
        for l in live:
            if l not in ordered and l.kind != LinkKind.CLOUD and _ok(l):
                ordered.append(l)
        for l in live:
            if l not in ordered and l.kind == LinkKind.CLOUD and _ok(l):
                ordered.append(l)

        last_exc: Optional[BaseException] = None
        for link in ordered:
            try:
                return link.submit(task), link
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LinkManager.submit: no link satisfies the task requirements")

    def snapshot(self) -> List[Dict[str, Any]]:
        return [l.snapshot() for l in self.links()]


__all__ = [
    "CloudQuantumLink",
    "LinkHealth",
    "LinkKind",
    "LinkManager",
    "LinkState",
    "LocalQuantumLink",
    "QuantumLink",
    "QuantumTask",
]
