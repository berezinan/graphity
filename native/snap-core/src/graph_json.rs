//! Structure scan of a node-link `graph.json` (change `streaming-snapshot-writer`, D1/D2).
//!
//! One sequential pass over the file finds the byte range of every top-level
//! member and, for top-level arrays, of every element. The caller parses
//! elements in batches with Python's `json.loads`, so the values it sees are
//! exactly what `json.load` of the whole file would give: a JSON element is
//! self-contained. The scan checks only structure — strings with their
//! escapes, nesting, the separators between top-level members and between
//! array elements, and the end of the document. What lies inside a value is
//! validated by `json.loads` when the value is read.
//!
//! The same pass hashes the bytes (sha256), and size / mtime come from the
//! same open handle, so the source recorded in a snapshot describes exactly
//! the bytes it was built from. `std::fs::File::open` shares read, write and
//! delete on Windows: a concurrent `os.replace` over the path succeeds, and
//! this handle keeps reading the old content.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::UNIX_EPOCH;

const BLOCK: usize = 8 << 20;

#[derive(Debug)]
pub enum GjError {
    Io(String),
    /// Not valid JSON, or not a JSON object at the top. Carries the message.
    Damaged(String),
}

impl std::fmt::Display for GjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GjError::Io(m) | GjError::Damaged(m) => f.write_str(m),
        }
    }
}

fn damaged(pos: u64, why: &str) -> GjError {
    GjError::Damaged(format!("graph.json is damaged at byte {pos}: {why}"))
}

fn io(e: std::io::Error) -> GjError {
    GjError::Io(format!("graph.json cannot be read: {e}"))
}

/// One top-level member: byte ranges `[start, end)` of its key token (with the
/// quotes) and of its value; for an array value, the flat element ranges
/// `[s0, e0, s1, e1, ...]`.
#[derive(Debug, Clone)]
pub struct Member {
    pub key: (u64, u64),
    pub value: (u64, u64),
    pub elems: Option<Vec<u64>>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Top {
    Start,
    AfterOpen,
    ExpectKey,
    InKey,
    AfterKey,
    ExpectValue,
    InValue,
    InArray,
    AfterValue,
    Done,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arr {
    AfterOpen,
    ExpectElem,
    InElem,
    AfterElem,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Skip {
    None,
    Str { esc: bool },
    Scalar,
    Container { depth: u32, in_str: bool, esc: bool },
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn start_skip(b: u8) -> Option<Skip> {
    match b {
        b'"' => Some(Skip::Str { esc: false }),
        b'{' | b'[' => Some(Skip::Container { depth: 1, in_str: false, esc: false }),
        // Numbers, true/false/null, and NaN / Infinity / -Infinity, which
        // Python's json writes and reads by default. The content of a scalar is
        // checked by json.loads when it is read.
        b'-' | b'0'..=b'9' | b't' | b'f' | b'n' | b'N' | b'I' => Some(Skip::Scalar),
        _ => None,
    }
}

/// Advance a value skipper over `buf[i..]`. Returns the next index and, when the
/// value ended, its exclusive end offset. A scalar ends BEFORE its delimiter,
/// which is not consumed.
fn skip(st: &mut Skip, buf: &[u8], mut i: usize, base: u64) -> (usize, Option<u64>) {
    match *st {
        Skip::None => (i, None),
        Skip::Str { mut esc } => {
            loop {
                if i >= buf.len() {
                    *st = Skip::Str { esc };
                    return (buf.len(), None);
                }
                if esc {
                    esc = false;
                    i += 1;
                    continue;
                }
                match memchr::memchr2(b'"', b'\\', &buf[i..]) {
                    None => {
                        *st = Skip::Str { esc: false };
                        return (buf.len(), None);
                    }
                    Some(k) => {
                        let j = i + k;
                        if buf[j] == b'\\' {
                            esc = true;
                            i = j + 1;
                        } else {
                            *st = Skip::None;
                            return (j + 1, Some(base + j as u64 + 1));
                        }
                    }
                }
            }
        }
        Skip::Scalar => {
            while i < buf.len() {
                match buf[i] {
                    b' ' | b'\t' | b'\n' | b'\r' | b',' | b']' | b'}' | b'[' | b'{' | b'"' | b':' => {
                        *st = Skip::None;
                        return (i, Some(base + i as u64));
                    }
                    _ => i += 1,
                }
            }
            (i, None)
        }
        Skip::Container { mut depth, mut in_str, mut esc } => {
            while i < buf.len() {
                if in_str {
                    if esc {
                        esc = false;
                        i += 1;
                        continue;
                    }
                    match memchr::memchr2(b'"', b'\\', &buf[i..]) {
                        None => {
                            i = buf.len();
                            break;
                        }
                        Some(k) => {
                            let j = i + k;
                            if buf[j] == b'\\' {
                                esc = true;
                            } else {
                                in_str = false;
                            }
                            i = j + 1;
                            continue;
                        }
                    }
                }
                match buf[i] {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            *st = Skip::None;
                            return (i + 1, Some(base + i as u64 + 1));
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            *st = Skip::Container { depth, in_str, esc };
            (i, None)
        }
    }
}

/// Push-style structure scanner; feed the file in any chunking.
pub struct Scanner {
    top: Top,
    arr: Arr,
    skip: Skip,
    key_start: u64,
    key_end: u64,
    value_start: u64,
    elem_start: u64,
    elems: Vec<u64>,
    members: Vec<Member>,
}

impl Default for Scanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Scanner {
    pub fn new() -> Self {
        Scanner {
            top: Top::Start,
            arr: Arr::AfterOpen,
            skip: Skip::None,
            key_start: 0,
            key_end: 0,
            value_start: 0,
            elem_start: 0,
            elems: Vec::new(),
            members: Vec::new(),
        }
    }

    fn value_ended(&mut self, end: u64) {
        match self.top {
            Top::InKey => {
                self.key_end = end;
                self.top = Top::AfterKey;
            }
            Top::InValue => {
                self.members.push(Member {
                    key: (self.key_start, self.key_end),
                    value: (self.value_start, end),
                    elems: None,
                });
                self.top = Top::AfterValue;
            }
            Top::InArray => {
                self.elems.push(self.elem_start);
                self.elems.push(end);
                self.arr = Arr::AfterElem;
            }
            _ => unreachable!("a value ended outside a value"),
        }
    }

    fn array_ended(&mut self, end: u64) {
        self.members.push(Member {
            key: (self.key_start, self.key_end),
            value: (self.value_start, end),
            elems: Some(std::mem::take(&mut self.elems)),
        });
        self.top = Top::AfterValue;
    }

    pub fn feed(&mut self, buf: &[u8], base: u64) -> Result<(), GjError> {
        let mut i = 0;
        while i < buf.len() {
            if self.skip != Skip::None {
                let (ni, end) = skip(&mut self.skip, buf, i, base);
                i = ni;
                if let Some(e) = end {
                    self.value_ended(e);
                }
                continue;
            }
            let b = buf[i];
            let p = base + i as u64;
            if is_ws(b) {
                i += 1;
                continue;
            }
            match self.top {
                Top::Start => {
                    if b != b'{' {
                        return Err(damaged(p, "expected '{' at the start of the document"));
                    }
                    self.top = Top::AfterOpen;
                }
                Top::AfterOpen | Top::ExpectKey => {
                    if b == b'"' {
                        self.key_start = p;
                        self.top = Top::InKey;
                        self.skip = Skip::Str { esc: false };
                    } else if b == b'}' && self.top == Top::AfterOpen {
                        self.top = Top::Done;
                    } else {
                        return Err(damaged(p, "expected a key"));
                    }
                }
                Top::AfterKey => {
                    if b != b':' {
                        return Err(damaged(p, "expected ':' after a key"));
                    }
                    self.top = Top::ExpectValue;
                }
                Top::ExpectValue => {
                    self.value_start = p;
                    if b == b'[' {
                        self.top = Top::InArray;
                        self.arr = Arr::AfterOpen;
                        self.elems = Vec::new();
                    } else {
                        self.skip = start_skip(b).ok_or_else(|| damaged(p, "expected a value"))?;
                        self.top = Top::InValue;
                    }
                }
                Top::InArray => match self.arr {
                    Arr::AfterOpen | Arr::ExpectElem => {
                        if b == b']' && self.arr == Arr::AfterOpen {
                            self.array_ended(p + 1);
                        } else {
                            self.skip = start_skip(b)
                                .ok_or_else(|| damaged(p, "expected an array element"))?;
                            self.elem_start = p;
                            self.arr = Arr::InElem;
                        }
                    }
                    Arr::AfterElem => {
                        if b == b',' {
                            self.arr = Arr::ExpectElem;
                        } else if b == b']' {
                            self.array_ended(p + 1);
                        } else {
                            return Err(damaged(p, "expected ',' or ']' between array elements"));
                        }
                    }
                    Arr::InElem => unreachable!("element without a skipper"),
                },
                Top::AfterValue => {
                    if b == b',' {
                        self.top = Top::ExpectKey;
                    } else if b == b'}' {
                        self.top = Top::Done;
                    } else {
                        return Err(damaged(p, "expected ',' or '}' between top-level members"));
                    }
                }
                Top::Done => return Err(damaged(p, "extra data after the closing '}'")),
                Top::InKey | Top::InValue => unreachable!("value state without a skipper"),
            }
            i += 1;
        }
        Ok(())
    }

    pub fn finish(self, size: u64) -> Result<Vec<Member>, GjError> {
        if self.top != Top::Done || self.skip != Skip::None {
            return Err(damaged(size, "the file ends before the document does (truncated)"));
        }
        Ok(self.members)
    }
}

/// A scanned graph.json, kept open for reading value and element ranges.
pub struct GraphJson {
    file: File,
    pub size: u64,
    pub mtime_ns: i64,
    pub sha256: [u8; 32],
    pub members: Vec<Member>,
}

fn mtime_ns(meta: &std::fs::Metadata) -> Result<i64, GjError> {
    let t = meta.modified().map_err(io)?;
    Ok(match t.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i64,
        Err(e) => -(e.duration().as_nanos() as i64),
    })
}

impl GraphJson {
    /// Pass 1: hash, scan, and record the source of exactly the bytes read.
    pub fn open(path: &Path) -> Result<GraphJson, GjError> {
        let mut file = File::open(path).map_err(io)?;
        let meta = file.metadata().map_err(io)?;
        let size = meta.len();
        let mtime = mtime_ns(&meta)?;
        let mut hasher = Sha256::new();
        let mut sc = Scanner::new();
        let mut buf = vec![0u8; BLOCK];
        let mut base = 0u64;
        loop {
            let n = file.read(&mut buf).map_err(io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            sc.feed(&buf[..n], base)?;
            base += n as u64;
        }
        if base != size {
            return Err(GjError::Io(format!(
                "graph.json changed while it was read ({size} bytes expected, {base} read)"
            )));
        }
        let members = sc.finish(base)?;
        Ok(GraphJson { file, size, mtime_ns: mtime, sha256: hasher.finalize().into(), members })
    }

    pub fn read_range(&mut self, start: u64, end: u64) -> Result<Vec<u8>, GjError> {
        let mut out = vec![0u8; (end - start) as usize];
        self.file.seek(SeekFrom::Start(start)).map_err(io)?;
        self.file.read_exact(&mut out).map_err(io)?;
        Ok(out)
    }

    fn member(&self, i: usize) -> Result<&Member, GjError> {
        self.members.get(i).ok_or_else(|| GjError::Io(format!("no top-level member {i}")))
    }

    pub fn key(&mut self, i: usize) -> Result<Vec<u8>, GjError> {
        let (s, e) = self.member(i)?.key;
        self.read_range(s, e)
    }

    pub fn value(&mut self, i: usize) -> Result<Vec<u8>, GjError> {
        let (s, e) = self.member(i)?.value;
        self.read_range(s, e)
    }

    /// Number of elements when member `i` is an array.
    pub fn count(&self, i: usize) -> Option<usize> {
        self.members.get(i)?.elems.as_ref().map(|v| v.len() / 2)
    }

    /// `[e_start, ..., e_{start+n-1}]` as one JSON array. The bytes between
    /// elements are commas and whitespace, already checked by the scan.
    pub fn batch(&mut self, i: usize, start: usize, n: usize) -> Result<Vec<u8>, GjError> {
        let elems = self
            .member(i)?
            .elems
            .as_ref()
            .ok_or_else(|| GjError::Io(format!("top-level member {i} is not an array")))?;
        let count = elems.len() / 2;
        let end = (start + n).min(count);
        if start >= end {
            return Ok(b"[]".to_vec());
        }
        let (s, e) = (elems[2 * start], elems[2 * (end - 1) + 1]);
        let mut out = Vec::with_capacity((e - s) as usize + 2);
        out.push(b'[');
        out.extend(self.read_range(s, e)?);
        out.push(b']');
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_chunked(doc: &[u8], chunk: usize) -> Result<Vec<Member>, GjError> {
        let mut sc = Scanner::new();
        for (k, c) in doc.chunks(chunk.max(1)).enumerate() {
            sc.feed(c, (k * chunk.max(1)) as u64)?;
        }
        sc.finish(doc.len() as u64)
    }

    fn slices<'a>(doc: &'a [u8], m: &Member) -> (&'a [u8], &'a [u8], Vec<&'a [u8]>) {
        let key = &doc[m.key.0 as usize..m.key.1 as usize];
        let val = &doc[m.value.0 as usize..m.value.1 as usize];
        let els = m
            .elems
            .as_ref()
            .map(|v| v.chunks(2).map(|p| &doc[p[0] as usize..p[1] as usize]).collect())
            .unwrap_or_default();
        (key, val, els)
    }

    /// Every chunking gives the same result as one chunk.
    fn scan(doc: &[u8]) -> Vec<Member> {
        let whole = scan_chunked(doc, doc.len()).expect("valid document");
        for chunk in [1, 2, 3, 7, 64] {
            let m = scan_chunked(doc, chunk).expect("valid document in chunks");
            assert_eq!(format!("{m:?}"), format!("{whole:?}"), "chunk {chunk}");
        }
        whole
    }

    #[test]
    fn indented_and_one_line_agree() {
        let indented = b"{\r\n  \"directed\": false,\r\n  \"nodes\": [\r\n    {\r\n      \"id\": \"a\"\r\n    },\r\n    {\r\n      \"id\": \"b\"\r\n    }\r\n  ],\r\n  \"links\": []\r\n}";
        let one = br#"{"directed":false,"nodes":[{"id":"a"},{"id":"b"}],"links":[]}"#;
        for doc in [&indented[..], &one[..]] {
            let m = scan(doc);
            assert_eq!(m.len(), 3);
            let (k, v, _) = slices(doc, &m[0]);
            assert_eq!((k, v), (&br#""directed""#[..], &b"false"[..]));
            let (k, _, els) = slices(doc, &m[1]);
            assert_eq!(k, br#""nodes""#);
            let els: Vec<String> = els.iter().map(|e| String::from_utf8(e.iter().copied().filter(|b| !is_ws(*b)).collect()).unwrap()).collect();
            assert_eq!(els, vec![r#"{"id":"a"}"#, r#"{"id":"b"}"#]);
            assert_eq!(m[2].elems.as_ref().unwrap().len(), 0);
        }
    }

    #[test]
    fn whitespace_anywhere() {
        let doc = b" \t\n{ \"nodes\" \n: [ 1 , {\"a\" : [ ] } ,\"s\" ] , \"x\":null }\n ";
        let m = scan(doc);
        let (_, _, els) = slices(doc, &m[0]);
        assert_eq!(els, vec![&b"1"[..], br#"{"a" : [ ] }"#, br#""s""#]);
        let (_, v, _) = slices(doc, &m[1]);
        assert_eq!(v, b"null");
    }

    #[test]
    fn escapes_do_not_end_strings_or_open_containers() {
        let doc = br#"{"nodes":[{"label":"q\" ] } [ {","p":"c:\\"},{"u":"\u005d\ud83d\ude00\\\""}],"k\"ey":"v"}"#;
        let m = scan(doc);
        let (_, _, els) = slices(doc, &m[0]);
        assert_eq!(els, vec![&br#"{"label":"q\" ] } [ {","p":"c:\\"}"#[..], br#"{"u":"\u005d\ud83d\ude00\\\""}"#]);
        let (k, v, _) = slices(doc, &m[1]);
        assert_eq!((k, v), (&br#""k\"ey""#[..], &br#""v""#[..]));
    }

    #[test]
    fn nested_values_in_attributes() {
        let doc = br#"{"graph":{"hyperedges":[{"nodes":["a","b"]}]},"links":[{"extra":[1,{"k":null}],"w":-1.5e3},{"n":NaN,"i":-Infinity}]}"#;
        let m = scan(doc);
        let (_, v, els) = slices(doc, &m[0]);
        assert!(els.is_empty() && m[0].elems.is_none());
        assert_eq!(v, br#"{"hyperedges":[{"nodes":["a","b"]}]}"#);
        let (_, _, els) = slices(doc, &m[1]);
        assert_eq!(els.len(), 2);
        assert_eq!(els[1], br#"{"n":NaN,"i":-Infinity}"#);
    }

    #[test]
    fn empty_object_and_arrays() {
        assert!(scan(b"{}").is_empty());
        let m = scan(br#"{"nodes":[],"edges":[ ]}"#);
        assert_eq!(m.iter().map(|x| x.elems.as_ref().unwrap().len()).collect::<Vec<_>>(), vec![0, 0]);
    }

    #[test]
    fn repeated_key_keeps_every_occurrence() {
        // The caller resolves repeats the way json.load does (last value wins).
        let doc = br#"{"directed":true,"nodes":[],"directed":false}"#;
        let m = scan(doc);
        assert_eq!(m.len(), 3);
        assert_eq!(slices(doc, &m[2]).1, b"false");
    }

    fn damaged_at(doc: &[u8]) -> String {
        match scan_chunked(doc, doc.len()) {
            Err(GjError::Damaged(m)) => m,
            other => panic!("expected a damage error, got {other:?}"),
        }
    }

    #[test]
    fn truncated_document_is_refused() {
        let doc = br#"{"nodes":[{"id":"a"},{"id":"b"}],"links":[{"source":"a","target":"b"},{"sou"#;
        for cut in [doc.len(), doc.len() - 5, 12, 1] {
            let m = damaged_at(&doc[..cut]);
            assert!(m.contains("truncated"), "{cut}: {m}");
            assert!(m.contains(&format!("byte {cut}")), "{cut}: {m}");
        }
    }

    #[test]
    fn bad_separators_are_refused_with_the_offset() {
        let cases: [(&[u8], u64, &str); 6] = [
            (br#"{"nodes":[{"id":"a"} {"id":"b"}]}"#, 21, "between array elements"),
            (br#"{"nodes":[{"id":"a"},,{"id":"b"}]}"#, 21, "array element"),
            (br#"{"nodes":[{"id":"a"},]}"#, 21, "array element"),
            (br#"{"nodes":[1] "links":[]}"#, 13, "top-level members"),
            (br#"{"nodes":[1]} x"#, 14, "extra data"),
            (br#"["nodes"]"#, 0, "start of the document"),
        ];
        for (doc, pos, why) in cases {
            let m = damaged_at(doc);
            assert!(m.contains(&format!("byte {pos}:")) && m.contains(why), "{}: {m}", String::from_utf8_lossy(doc));
        }
    }

    #[test]
    fn open_reads_source_and_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("graph.json");
        let doc = br#"{"directed":false,"nodes":[{"id":"a"},{"id":"b"},{"id":"c"}],"links":[]}"#;
        std::fs::write(&p, doc).unwrap();
        let mut g = GraphJson::open(&p).unwrap();
        assert_eq!(g.size, doc.len() as u64);
        assert_eq!(g.sha256, <[u8; 32]>::from(Sha256::digest(doc)));
        let meta = std::fs::metadata(&p).unwrap();
        assert_eq!(g.mtime_ns, mtime_ns(&meta).unwrap());
        assert_eq!(g.key(1).unwrap(), br#""nodes""#);
        assert_eq!(g.count(1), Some(3));
        assert_eq!(g.count(0), None);
        assert_eq!(g.batch(1, 1, 5).unwrap(), br#"[{"id":"b"},{"id":"c"}]"#);
        assert_eq!(g.batch(1, 3, 2).unwrap(), b"[]");
        assert_eq!(g.value(0).unwrap(), b"false");
    }

    #[test]
    fn path_can_be_replaced_while_open() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("graph.json");
        std::fs::write(&p, br#"{"nodes":[{"id":"old"}]}"#).unwrap();
        let mut g = GraphJson::open(&p).unwrap();
        let tmp = dir.path().join("graph.json.new");
        std::fs::write(&tmp, br#"{"nodes":[{"id":"new"}]}"#).unwrap();
        std::fs::rename(&tmp, &p).unwrap();
        assert_eq!(g.batch(0, 0, 1).unwrap(), br#"[{"id":"old"}]"#);
    }
}
