"""ArcadeDB as a machine service: install / uninstall / status.

graphify does not run the database, the operating system does. This module owns
everything about that boundary: the machine-wide layout under ProgramData, the
pinned distribution, the Windows service definition, and the machine config
every client reads to find the service. It deliberately has no start/stop — the
service control manager owns the process, and a second way to launch one would
mean a second hive and a second configuration.

The pure helpers (paths, URLs, the java argv, the service XML) are unit-testable;
download and service registration are thin I/O. Point ``GRAPHIFY_ARCADE_HOME``
at an existing layout to relocate the whole thing.
"""
from __future__ import annotations

import os
import re
import subprocess
import tarfile
from pathlib import Path

SERVICE_NAME = "GraphifyArcadeDB"  # the Windows service `arcade install-service` registers


class GraphDBUnavailable(RuntimeError):
    """The graph database is unreachable, or holds no database for this project.

    Carries the complete user-facing text: the graph database is required, so
    there is no fallback to weigh up and the only decision left to the caller is
    where to print it. Defined in this stdlib-only module so the CLI entry point
    can catch it without importing the query stack (networkx and friends) on
    every command.
    """


ARCADE_VERSION = "26.6.1"
_DEFAULT_HEAP = "2G"
_MIN_JAVA = 21  # ArcadeDB 26.x class files are Java 21
# JVM -Xmx grammar: digits with an optional K/M/G unit. Enforced because the value
# is interpolated straight into the java argv.
_HEAP_RE = re.compile(r"^[0-9]+[KkMmGg]?$")


def default_heap() -> str:
    """Heap for the ArcadeDB JVM — ``GRAPHIFY_ARCADE_HEAP``, else 2G.

    The 2G default is sized for ordinary corpora. A graph in the million-node
    range does not fit, and ArcadeDB does not fail loudly when it doesn't: the
    server keeps answering from memory while bucket files stay empty on disk and
    index lookups silently miss, so the database looks alive and is neither
    complete nor persisted. Raising the heap is the fix, hence the override.

    Read at call time, not import time, so setting the variable in-process (or
    after the module is imported) takes effect.
    """
    raw = (os.environ.get("GRAPHIFY_ARCADE_HEAP") or "").strip()
    if not raw:
        return _DEFAULT_HEAP
    if not _HEAP_RE.match(raw):
        raise ValueError(
            f"GRAPHIFY_ARCADE_HEAP={raw!r} is not a JVM heap size — expected digits "
            "with an optional K/M/G unit (e.g. 8G, 512M). Refusing to pass it to java."
        )
    return raw


def arcade_home() -> Path:
    """Machine-wide install root (override with ``GRAPHIFY_ARCADE_HOME``).

    Under ProgramData, not a user profile: one service serves every session on
    the machine, so its data cannot live in the profile of whoever happened to
    install it. Data, not just program files, hence ProgramData rather than
    Program Files — the default ACLs there let every user read the config
    without touching the permissions of the program directory.
    """
    env = os.environ.get("GRAPHIFY_ARCADE_HOME")
    if env:
        return Path(env)
    return Path(os.environ.get("PROGRAMDATA") or r"C:\ProgramData") / "graphify" / "arcadedb"


def dist_dir(home: Path | None = None) -> Path:
    """The extracted distribution — the one directory an upgrade replaces."""
    return (home or arcade_home()) / "dist" / f"arcadedb-{ARCADE_VERSION}"


def databases_dir(home: Path | None = None) -> Path:
    """The hive, deliberately OUTSIDE the version-named distribution directory.

    ArcadeDB defaults it to ``<rootPath>/databases``, i.e. inside the unpacked
    release, so bumping ARCADE_VERSION would orphan every database on the
    machine. Passed to the server explicitly (see `server_command`).
    """
    return (home or arcade_home()) / "databases"


def log_dir(home: Path | None = None) -> Path:
    return (home or arcade_home()) / "log"


def log_file(home: Path | None = None) -> Path:
    return log_dir(home) / "server.log"


def config_path(home: Path | None = None) -> Path:
    """Machine config read by every client: service address and shared password."""
    return (home or arcade_home()) / "service.json"


def service_exe(home: Path | None = None) -> Path:
    return (home or arcade_home()) / f"{SERVICE_NAME}.exe"


def service_xml(home: Path | None = None) -> Path:
    return (home or arcade_home()) / f"{SERVICE_NAME}.xml"


def read_machine_config(home: Path | None = None) -> dict:
    """The machine config every client reads, or ``{}`` when none is installed.

    Missing is normal (nobody has run ``install-service`` yet, or the caller is
    on the JSON backend). Present-but-unreadable is not: silently falling back
    to defaults there would report "cannot connect" while the real problem is a
    corrupted file, so that case is loud.
    """
    import json
    p = config_path(home)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GraphDBUnavailable(
            f"the machine config {p} could not be read: {exc}\n"
            f"Reinstall the service with: graphify arcade install-service") from exc
    return data if isinstance(data, dict) else {}


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


def server_command(password: str, *, java_exe: str | None = None, heap: str | None = None,
                   port: int = 2480, home: Path | None = None) -> list[str]:
    """The java argv to launch ArcadeDB. Run with cwd = dist_dir so the relative
    ``lib/*`` classpath (Java expands the wildcard itself, sidestepping the
    Unix-vs-Windows path-separator bug in the bundled server.sh) and ``config/``
    resolve. ``port`` binds the HTTP listener so a non-default GRAPHIFY_ARCADE_URL
    actually takes effect. The database directory is passed explicitly so the
    hive survives a distribution upgrade (see `databases_dir`)."""
    return [
        java_exe or java_executable(),
        f"-Xms512M", f"-Xmx{heap or default_heap()}",
        "--add-opens", "java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens", "java.base/java.nio.channels.spi=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Djava.awt.headless=true", "-Dfile.encoding=UTF8",
        "-Djava.util.logging.config.file=config/arcadedb-log.properties",
        f"-Darcadedb.server.rootPassword={password}",
        f"-Darcadedb.server.httpIncomingPort={port}",
        f"-Darcadedb.server.databaseDirectory={databases_dir(home)}",
        "-cp", "lib/*",
        "com.arcadedb.server.ArcadeDBServer",
    ]




def status(host: str = "127.0.0.1", port: int = 2480, password: str | None = None,
           home: Path | None = None) -> dict:
    """Whether the service answers on HTTP, plus how the OS sees the service.

    The password defaults to the machine config rather than a constant: with a
    shared service there is one credential per machine and no reason for the
    caller to have to know it.
    """
    if password is None:
        password = read_machine_config(home).get("password", "")
    ready = False
    try:
        import requests
        r = requests.get(f"http://{host}:{port}/api/v1/ready", auth=("root", password), timeout=3)
        ready = r.status_code in (200, 204)
    except Exception:
        ready = False
    return {"ready": ready, "service": service_state(), "installed": is_installed(home),
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
    target = dist_dir(home).parent
    target.mkdir(parents=True, exist_ok=True)
    archive = home / "arcadedb.tar.gz"
    with requests.get(download_url(), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target)
    archive.unlink(missing_ok=True)
    if not is_installed(home):
        raise RuntimeError(f"ArcadeDB install incomplete under {home}")
    return dist_dir(home)


# --------------------------------------------------------------------------- #
# Windows service
# --------------------------------------------------------------------------- #
# WinSW wraps a plain process as a Windows service. `sc.exe create` cannot: it
# registers a service-aware executable that answers the SCM, and java.exe is not
# one. Pinned and fetched the way the distribution already is, so the Python
# package still ships no binary of its own.
WINSW_VERSION = "2.12.0"


def winsw_url(version: str = WINSW_VERSION) -> str:
    # `WinSW.NET461.exe` is how the 2.x line names the .NET Framework build; the
    # hyphenated `WinSW-net461.exe` belongs to the 3.x alphas and 404s here.
    return (f"https://github.com/winsw/winsw/releases/download/"
            f"v{version}/WinSW.NET461.exe")


def download_winsw(home: Path | None = None) -> Path:
    """Fetch the pinned service wrapper (no-op if already there)."""
    home = home or arcade_home()
    exe = service_exe(home)
    if exe.is_file():
        return exe
    try:
        import requests
    except ImportError as e:
        raise ImportError('Installing the service needs requests. Run: pip install "graphifyy[arcadedb]"') from e
    home.mkdir(parents=True, exist_ok=True)
    with requests.get(winsw_url(), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(exe, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return exe


def is_admin() -> bool:
    """True when running elevated. Registering a service needs it."""
    if os.name != "nt":
        return hasattr(os, "geteuid") and os.geteuid() == 0
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def generate_password(nbytes: int = 24) -> str:
    """A per-machine credential, generated at install and stored in the config.

    Not a constant in the source: a personal, short-lived server could afford a
    well-known password, a permanent service shared by every session on the
    machine cannot. It is still not an access boundary between those users --
    the hive is shared on purpose -- it just is not public knowledge.
    """
    import secrets
    return secrets.token_urlsafe(nbytes)


def write_machine_config(home: Path | None = None, *, url: str, password: str,
                         user: str = "root") -> Path:
    """Write the config every client on this machine reads."""
    import json
    home = home or arcade_home()
    home.mkdir(parents=True, exist_ok=True)
    p = config_path(home)
    p.write_text(json.dumps({"url": url, "user": user, "password": password,
                             "home": str(home)}, indent=2), encoding="utf-8")
    return p


def service_definition(*, java_exe: str, password: str, port: int, heap: str | None = None,
                       home: Path | None = None) -> str:
    """The WinSW service definition XML.

    ``java_exe`` is an ABSOLUTE path resolved at install time on purpose: the
    service runs under an account that has no JAVA_HOME, so a definition leaning
    on the installing user's environment would simply fail to start at boot --
    and the client would then report the database as unreachable, naming the
    wrong cause.
    """
    from xml.sax.saxutils import escape
    home = home or arcade_home()
    argv = server_command(password, java_exe=java_exe, heap=heap, port=port, home=home)
    args = " ".join(f'"{a}"' if " " in a else a for a in argv[1:])
    onfailure = '  <onfailure action="restart" delay="10 sec"/>'
    return "\n".join([
        "<service>",
        f"  <id>{SERVICE_NAME}</id>",
        "  <name>Graphify ArcadeDB</name>",
        "  <description>Graph database serving every graphify session on this machine.</description>",
        f"  <executable>{escape(java_exe)}</executable>",
        f"  <arguments>{escape(args)}</arguments>",
        f"  <workingdirectory>{escape(str(dist_dir(home)))}</workingdirectory>",
        f"  <logpath>{escape(str(log_dir(home)))}</logpath>",
        "  <logmode>roll</logmode>",
        "  <startmode>Automatic</startmode>",
        onfailure,
        "</service>",
        "",
    ])


def service_state() -> str:
    """What the SCM says: 'running', 'stopped', 'not installed' or 'unknown'."""
    if os.name != "nt":
        return "not installed"
    try:
        out = subprocess.run(["sc", "query", SERVICE_NAME], capture_output=True,
                             text=True, timeout=15)
    except Exception:
        return "unknown"
    if out.returncode != 0:
        return "not installed"
    text = out.stdout.upper()
    if "RUNNING" in text:
        return "running"
    if "STOPPED" in text or "PENDING" in text:
        return "stopped"
    return "unknown"


def server_users_file(home: Path | None = None) -> Path:
    """Where ArcadeDB keeps the credentials it created on first start."""
    return dist_dir(home) / "config" / "server-users.jsonl"


def _existing_password_differs(password: str, home: Path | None = None) -> bool:
    """True when the hive already holds credentials this password will not open.

    ``-Darcadedb.server.rootPassword`` only takes effect while the security file
    is being created; once ``server-users.jsonl`` exists the server keeps the old
    credential. Installing over it would leave a running service whose password
    is not the one in the machine config -- a service that looks healthy and
    answers nobody.
    """
    if not server_users_file(home).is_file():
        return False
    return read_machine_config(home).get("password") != password


def install_service(*, password: str | None = None, port: int = 2480, heap: str | None = None,
                    home: Path | None = None) -> dict:
    """Install ArcadeDB as an auto-starting Windows service.

    Order matters: elevation is checked FIRST, so an unprivileged run costs
    nothing and leaves no half-finished install behind.
    """
    if os.name != "nt":
        raise RuntimeError("install-service is Windows-only.")
    if not is_admin():
        raise RuntimeError(
            "installing a service needs an elevated prompt. Re-run "
            "`graphify arcade install-service` from an administrator terminal.")

    home = home or arcade_home()
    password = password or read_machine_config(home).get("password") or generate_password()
    if _existing_password_differs(password, home):
        raise RuntimeError(
            "this hive already has server credentials and the given password is not the "
            "one in the machine config. ArcadeDB honours the root password only while "
            "creating its security file, so installing now would leave a service whose "
            "password nothing knows.\n"
            "To change it: graphify arcade uninstall-service, delete "
            f"{server_users_file(home)}, then install again.")

    for d in (home, dist_dir(home).parent, databases_dir(home), log_dir(home)):
        d.mkdir(parents=True, exist_ok=True)
    download(home)
    java_exe = str(Path(pick_java()).resolve())  # absolute: the service account has no JAVA_HOME
    exe = download_winsw(home)
    service_xml(home).write_text(
        service_definition(java_exe=java_exe, password=password, port=port, heap=heap, home=home),
        encoding="utf-8")
    write_machine_config(home, url=f"http://127.0.0.1:{port}", password=password)

    subprocess.run([str(exe), "install"], check=True, capture_output=True, text=True)
    subprocess.run([str(exe), "start"], check=True, capture_output=True, text=True)
    return {"home": str(home), "service": SERVICE_NAME, "url": f"http://127.0.0.1:{port}",
            "java": java_exe, "databases": str(databases_dir(home))}


def uninstall_service(home: Path | None = None) -> bool:
    """Stop and deregister the service. The hive of databases is left alone."""
    home = home or arcade_home()
    exe = service_exe(home)
    if not exe.is_file():
        return False
    subprocess.run([str(exe), "stop"], capture_output=True, text=True)
    subprocess.run([str(exe), "uninstall"], check=True, capture_output=True, text=True)
    return True
