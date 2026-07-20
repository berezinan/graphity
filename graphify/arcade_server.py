"""Lifecycle helper for a local ArcadeDB server (download / start / stop / status).

Connect-only is the default everywhere else; this module is the opt-in
convenience that makes the ArcadeDB backend turnkey — `graphify arcade start`
downloads a pinned ArcadeDB (once, cached) and launches it with the JVM flags it
needs, so users never type a raw `java -cp` line.

The pure helpers (paths, download URL, the java argv) are unit-testable; the
download/launch/kill are thin I/O. Point ``GRAPHIFY_ARCADE_HOME`` at an existing
install to skip the download.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

ARCADE_VERSION = "26.6.1"
_DEFAULT_HEAP = "2G"
_MIN_JAVA = 21  # ArcadeDB 26.x class files are Java 21


def arcade_home() -> Path:
    """Install root holding the extracted distribution (override with env)."""
    env = os.environ.get("GRAPHIFY_ARCADE_HOME")
    return Path(env) if env else Path.home() / ".graphify" / "arcadedb"


def dist_dir(home: Path | None = None) -> Path:
    return (home or arcade_home()) / f"arcadedb-{ARCADE_VERSION}"


def pid_file(home: Path | None = None) -> Path:
    return (home or arcade_home()) / "server.pid"


def log_file(home: Path | None = None) -> Path:
    return (home or arcade_home()) / "server.log"


def download_url(version: str = ARCADE_VERSION) -> str:
    return (f"https://github.com/ArcadeData/arcadedb/releases/download/"
            f"{version}/arcadedb-{version}-minimal.tar.gz")


def is_installed(home: Path | None = None) -> bool:
    """True if the server jar is present in the distribution's lib dir."""
    lib = dist_dir(home) / "lib"
    return lib.is_dir() and any(p.name.startswith("arcadedb-server-") for p in lib.glob("*.jar"))


def java_executable() -> str:
    jh = os.environ.get("JAVA_HOME")
    if jh:
        exe = Path(jh) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if exe.exists():
            return str(exe)
    return "java"


def _java_major(exe: str) -> int | None:
    """Major Java version reported by ``exe -version`` (None if it can't run)."""
    import re
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stderr
    except Exception:
        return None
    m = re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else None


def pick_java() -> str:
    """Choose a Java >= _MIN_JAVA: prefer JAVA_HOME's, else PATH's ``java``.

    Raises a clear error rather than letting the JVM fail with a cryptic
    UnsupportedClassVersionError when only an older JDK is available (a common
    case when JAVA_HOME points at an LTS that predates ArcadeDB's baseline)."""
    seen = None
    for exe in (java_executable(), "java"):
        v = _java_major(exe)
        if v is not None and v >= _MIN_JAVA:
            return exe
        if v is not None:
            seen = v
    raise RuntimeError(
        f"ArcadeDB needs Java >= {_MIN_JAVA}, but the available Java is "
        f"{seen if seen is not None else 'missing'}. Point JAVA_HOME at a newer JDK."
    )


def server_command(password: str, *, java_exe: str | None = None, heap: str = _DEFAULT_HEAP,
                   port: int = 2480) -> list[str]:
    """The java argv to launch ArcadeDB. Run with cwd = dist_dir so the relative
    ``lib/*`` classpath (Java expands the wildcard itself, sidestepping the
    Unix-vs-Windows path-separator bug in the bundled server.sh) and ``config/``
    resolve. ``port`` binds the HTTP listener so a non-default GRAPHIFY_ARCADE_URL
    actually takes effect."""
    return [
        java_exe or java_executable(),
        f"-Xms512M", f"-Xmx{heap}",
        "--add-opens", "java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens", "java.base/java.nio.channels.spi=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Djava.awt.headless=true", "-Dfile.encoding=UTF8",
        "-Djava.util.logging.config.file=config/arcadedb-log.properties",
        f"-Darcadedb.server.rootPassword={password}",
        f"-Darcadedb.server.httpIncomingPort={port}",
        "-cp", "lib/*",
        "com.arcadedb.server.ArcadeDBServer",
    ]


def _read_pid(home: Path | None = None) -> int | None:
    pf = pid_file(home)
    if pf.exists():
        try:
            return int(pf.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    return None


def status(host: str = "127.0.0.1", port: int = 2480, password: str = "playwithdata",
           home: Path | None = None) -> dict:
    """Whether the server answers on HTTP, plus the tracked PID (if any)."""
    ready = False
    try:
        import requests
        r = requests.get(f"http://{host}:{port}/api/v1/ready", auth=("root", password), timeout=3)
        ready = r.status_code in (200, 204)
    except Exception:
        ready = False
    return {"ready": ready, "pid": _read_pid(home), "installed": is_installed(home),
            "home": str(arcade_home() if home is None else home)}


def download(home: Path | None = None) -> Path:
    """Fetch + extract the pinned distribution (no-op if already installed)."""
    home = home or arcade_home()
    if is_installed(home):
        return dist_dir(home)
    try:
        import requests
    except ImportError as e:
        raise ImportError('Downloading ArcadeDB needs requests. Run: pip install "graphifyy[arcadedb]"') from e
    home.mkdir(parents=True, exist_ok=True)
    archive = home / "arcadedb.tar.gz"
    with requests.get(download_url(), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(home)
    archive.unlink(missing_ok=True)
    if not is_installed(home):
        raise RuntimeError(f"ArcadeDB install incomplete under {home}")
    return dist_dir(home)


def start(password: str = "playwithdata", *, host: str = "127.0.0.1", port: int = 2480,
          heap: str = _DEFAULT_HEAP, home: Path | None = None) -> dict:
    """Download if needed, then launch a detached server. No-op if already up."""
    home = home or arcade_home()
    st = status(host, port, password, home)
    if st["ready"]:
        return {"already_running": True, **st}
    download(home)
    java_exe = pick_java()  # fail clearly if no adequate Java rather than via the JVM
    log = log_file(home)
    log.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    env = {**os.environ, "ARCADEDB_SERVER_ROOT_PASSWORD": password}
    with open(log, "ab") as logf:
        proc = subprocess.Popen(
            server_command(password, java_exe=java_exe, heap=heap, port=port), cwd=str(dist_dir(home)),
            stdout=logf, stderr=logf, stdin=subprocess.DEVNULL, env=env, **kwargs,
        )
    pid_file(home).write_text(str(proc.pid), encoding="utf-8")
    return {"already_running": False, "pid": proc.pid, "log": str(log),
            "url": f"http://{host}:{port}"}


_AUTOSTART_READY_TIMEOUT = 60.0  # seconds to wait for the launched server's /ready


def autostart_enabled() -> bool:
    """True when GRAPHIFY_ARCADE_AUTOSTART is truthy (opt-in, off by default)."""
    return os.environ.get("GRAPHIFY_ARCADE_AUTOSTART", "").strip().lower() in ("1", "true", "yes")


def maybe_autostart(cfg: dict) -> None:
    """Opt-in auto-start of the local server before an extract/update sync.

    No-op unless GRAPHIFY_ARCADE_AUTOSTART is truthy. Reuses the resolved
    backend config (host/port/password from GRAPHIFY_ARCADE_URL /
    GRAPHIFY_ARCADE_PASSWORD) so auto-start and connect-only point at the same
    server. Any failure (no Java, download error, ready-timeout) is swallowed
    with a warning — the pipeline then degrades to the JSON graph exactly as
    connect-only would.

    Lives here, NOT in the CLI dispatch, so upstream merges over churn files
    can only sever a one-line call site — a loss the tracked guard test makes
    visible (see openspec spec `arcadedb-backend`).
    """
    if not autostart_enabled():
        return
    host = cfg.get("host") or "127.0.0.1"
    port = int(cfg.get("port") or 2480)
    password = cfg.get("password") or os.environ.get("GRAPHIFY_ARCADE_PASSWORD") or "playwithdata"
    try:
        if status(host, port, password).get("ready"):
            return
        start(password, host=host, port=port)
        import time
        deadline = time.monotonic() + _AUTOSTART_READY_TIMEOUT
        while True:
            if status(host, port, password).get("ready"):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"server not ready within {_AUTOSTART_READY_TIMEOUT:.0f}s "
                    f"(see {log_file()})"
                )
            time.sleep(1.0)
    except Exception as exc:
        print(f"[graphify db] warning: ArcadeDB auto-start failed: {exc}", file=sys.stderr)


def stop(home: Path | None = None) -> bool:
    """Terminate the tracked server process. Returns True if a PID was signalled."""
    pid = _read_pid(home)
    if pid is None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    pid_file(home).unlink(missing_ok=True)
    return True
