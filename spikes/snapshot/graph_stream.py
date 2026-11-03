"""Streaming reader for an indent=2 graph.json (task 6.5).

`json.load` of an 11 GB graph.json peaks at about 3x the file (the bytes,
the decoded str and the objects coexist: 12.3 GB for Buh30's 4 GB file).
`graph.json` is written by `json.dump(..., indent=2)`, so every element of
the top-level `nodes` / `links` arrays starts on a line `    {` and ends on a
line `    }` or `    },`; JSON strings cannot hold a raw newline. Each element
is parsed on its own, so the peak is the objects plus one element.

`load(path)` returns exactly what `json.load` returns; a file not in that
layout falls back to `json.load`.
"""
from __future__ import annotations

import json

_TOP = b'  "'


def _top_key(line: bytes) -> tuple[str, bytes] | None:
    if not line.startswith(_TOP) or line.startswith(b'   '):
        return None
    end = line.index(b'":', 3)
    return json.loads(line[2:end + 1]), line[end + 2:].strip()


def load(path) -> dict:
    with open(path, "rb") as f:
        first = f.readline()
        second = f.readline()
        if first.rstrip(b"\r\n") != b"{" or not second.startswith(_TOP):
            f.seek(0)
            return json.loads(f.read())
        out: dict = {}
        line = second
        while line:
            stripped = line.rstrip(b"\r\n")
            if stripped == b"}":
                break
            kv = _top_key(stripped)
            if kv is None:
                raise ValueError(f"unexpected top-level line: {stripped[:80]!r}")
            key, rest = kv
            if rest.endswith(b","):
                rest = rest[:-1]
            if rest in (b"[", b"{"):
                closer = b"  ]" if rest == b"[" else b"  }"
                if key in ("nodes", "links", "edges") and rest == b"[":
                    items = []
                    buf: list[bytes] = []
                    for line in f:
                        s = line.rstrip(b"\r\n")
                        if s in (closer, closer + b","):
                            break
                        buf.append(s)
                        if s in (b"    }", b"    },"):
                            if buf[-1].endswith(b","):
                                buf[-1] = buf[-1][:-1]
                            items.append(json.loads(b"\n".join(buf)))
                            buf = []
                    else:
                        raise ValueError(f"unterminated array {key!r}")
                    if buf:
                        raise ValueError(f"trailing lines in array {key!r}")
                    out[key] = items
                else:
                    buf = [rest]
                    for line in f:
                        s = line.rstrip(b"\r\n")
                        if s in (closer, closer + b","):
                            buf.append(closer.strip())
                            break
                        buf.append(s)
                    else:
                        raise ValueError(f"unterminated value {key!r}")
                    out[key] = json.loads(b"\n".join(buf))
            else:
                out[key] = json.loads(rest)
            line = f.readline()
        else:
            raise ValueError("graph.json ended before its closing brace")
    return out
