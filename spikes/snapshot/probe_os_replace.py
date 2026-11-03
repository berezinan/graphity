"""D7 follow-up: can the Python writer `os.replace` over a file that a Rust
reader (memmap2) holds mapped in another process?

Usage: python probe_os_replace.py <winfs_probe.exe> <scratch-dir>
"""

import os
import subprocess
import sys
from pathlib import Path

exe, scratch = sys.argv[1], Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
live, tmp = scratch / "py.graph.snap", scratch / "py.graph.snap.tmp"
live.write_bytes(b"A" * 65536)

holder = subprocess.Popen(
    [exe, "hold", str(live)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)
assert holder.stdout.readline().strip() == "ready"

tmp.write_bytes(b"B" * 4096)
try:
    os.replace(tmp, live)
    result = "ok"
except OSError as e:
    result = f"ERR winerror={e.winerror} {e.strerror}"

holder_view = holder.communicate("\n")[0].strip()
now = live.read_bytes()
print(
    f"| os.replace over Rust-mapped file (other process) | {result} | "
    f"holder view {holder_view} | live name now {len(now)} B "
    f"{'new' if now[:1] == b'B' else 'old'} |"
)
for p in (live, tmp):
    try:
        p.unlink()
    except FileNotFoundError:
        pass
