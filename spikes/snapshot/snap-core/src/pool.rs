//! String pools (D3, D13).
//!
//! ```text
//! blob: \0 s0 \0 s1 \0 ... s(k-1) \0        off: [start of s0, ..., start of s(k-1), blob.len()]
//! ```
//!
//! String `j` is `blob[off[j] .. off[j+1] - 1]`. Every string sits between NUL
//! bytes, so one SIMD `memmem` pass over the blob answers substring (`needle`),
//! prefix (`\0needle`) and equality (`\0needle\0`) for every string at once.
//! A search pool also carries postings: string -> node indices, ascending.

use crate::header::SnapError;
use memchr::memmem;
use rustc_hash::FxHashMap;

pub struct PoolBuilder {
    pub blob: Vec<u8>,
    pub off: Vec<u64>,
    map: Option<FxHashMap<Box<[u8]>, u32>>,
}

impl PoolBuilder {
    pub fn new(dedup: bool) -> Self {
        PoolBuilder { blob: vec![0], off: vec![1], map: dedup.then(FxHashMap::default) }
    }

    pub fn len(&self) -> usize {
        self.off.len() - 1
    }

    /// Index of `s`, adding it unless the pool dedups and already holds it.
    pub fn push(&mut self, s: &[u8]) -> Result<u32, SnapError> {
        if let Some(m) = &self.map {
            if let Some(&j) = m.get(s) {
                return Ok(j);
            }
        }
        if memchr::memchr(0, s).is_some() {
            return Err(SnapError::Corrupt("string contains NUL".into()));
        }
        let j = self.len() as u32;
        self.blob.extend_from_slice(s);
        self.blob.push(0);
        self.off.push(self.blob.len() as u64);
        if let Some(m) = &mut self.map {
            m.insert(s.into(), j);
        }
        Ok(j)
    }

    pub fn get(&self, j: usize) -> &[u8] {
        &self.blob[self.off[j] as usize..self.off[j + 1] as usize - 1]
    }

    /// Release the dedup map once all strings are in.
    pub fn seal(&mut self) {
        self.map = None;
    }
}

/// Front-coded strings in insertion order (ids: neighbours in graph.json
/// order share long prefixes). Every `FC_BLOCK`-th string restarts in full.
pub const FC_BLOCK: usize = 16;

fn put_varint(v: &mut Vec<u8>, mut x: usize) {
    while x >= 0x80 {
        v.push((x as u8) | 0x80);
        x >>= 7;
    }
    v.push(x as u8);
}

fn get_varint(b: &[u8], pos: &mut usize) -> usize {
    let (mut x, mut shift) = (0usize, 0);
    loop {
        let byte = b[*pos];
        *pos += 1;
        x |= ((byte & 0x7f) as usize) << shift;
        if byte < 0x80 {
            return x;
        }
        shift += 7;
    }
}

pub fn front_code(p: &PoolBuilder) -> (Vec<u8>, Vec<u64>) {
    let mut blob = Vec::new();
    let mut off = Vec::with_capacity(p.len() / FC_BLOCK + 1);
    let mut prev: &[u8] = b"";
    for j in 0..p.len() {
        let s = p.get(j);
        let shared = if j % FC_BLOCK == 0 {
            off.push(blob.len() as u64);
            0
        } else {
            prev.iter().zip(s).take_while(|(a, b)| a == b).count()
        };
        put_varint(&mut blob, shared);
        put_varint(&mut blob, s.len() - shared);
        blob.extend_from_slice(&s[shared..]);
        prev = s;
    }
    (blob, off)
}

#[derive(Clone, Copy)]
pub struct FcPool<'a> {
    pub blob: &'a [u8],
    pub off: &'a [u64],
}

impl<'a> FcPool<'a> {
    pub fn get(&self, j: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut pos = self.off[j / FC_BLOCK] as usize;
        for _ in 0..=j % FC_BLOCK {
            let shared = get_varint(self.blob, &mut pos);
            let len = get_varint(self.blob, &mut pos);
            buf.truncate(shared);
            buf.extend_from_slice(&self.blob[pos..pos + len]);
            pos += len;
        }
        buf
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Substring,
    Prefix,
    Equal,
    /// The string is the needle followed only by `(` / `)`:
    /// `s.rstrip("()") == needle` for a needle that does not end in a paren.
    Bare,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        Some(match s {
            "substring" => Mode::Substring,
            "prefix" => Mode::Prefix,
            "equal" => Mode::Equal,
            "bare" => Mode::Bare,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
pub struct Pool<'a> {
    pub blob: &'a [u8],
    pub off: &'a [u64],
}

impl<'a> Pool<'a> {
    pub fn len(&self) -> usize {
        self.off.len().saturating_sub(1)
    }

    pub fn get(&self, j: usize) -> &'a [u8] {
        &self.blob[self.off[j] as usize..self.off[j + 1] as usize - 1]
    }

    /// String whose bytes contain position `pos` (`pos` inside the blob, not on the leading NUL).
    fn which(&self, pos: usize) -> usize {
        self.off.partition_point(|&o| (o as usize) <= pos) - 1
    }

    /// Ascending, deduplicated indices of the strings matching `needle`.
    pub fn matching(&self, needle: &[u8], mode: Mode) -> Vec<u32> {
        let n = self.len();
        if needle.is_empty() && matches!(mode, Mode::Substring | Mode::Prefix) {
            return (0..n as u32).collect();
        }
        let mut pat = Vec::with_capacity(needle.len() + 2);
        if mode != Mode::Substring {
            pat.push(0);
        }
        pat.extend_from_slice(needle);
        if mode == Mode::Equal {
            pat.push(0);
        }
        let finder = memmem::Finder::new(&pat);
        let mut out: Vec<u32> = Vec::new();
        let mut start = 0usize;
        while let Some(p) = finder.find(&self.blob[start..]) {
            let p = start + p;
            let j = match mode {
                Mode::Substring => self.which(p),
                // The match starts on the NUL in front of the string.
                _ => self.which(p + 1),
            };
            let keep = j < n
                && (mode != Mode::Bare
                    || self.get(j)[needle.len()..].iter().all(|&b| b == b'(' || b == b')'));
            if keep && out.last() != Some(&(j as u32)) {
                out.push(j as u32);
            }
            // Prefix / equality patterns start with the separator they share
            // with the previous string's end, so step one byte only.
            start = match mode {
                Mode::Substring => (self.off[j + 1] as usize).max(p + 1),
                _ => p + 1,
            };
            if start >= self.blob.len() {
                break;
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct Postings<'a> {
    pub off: &'a [u64],
    pub nodes: &'a [u32],
}

impl<'a> Postings<'a> {
    pub fn of(&self, j: usize) -> &'a [u32] {
        &self.nodes[self.off[j] as usize..self.off[j + 1] as usize]
    }
}

/// CSR from `(key, value)` pairs given in value order: per key, values ascending.
pub fn csr(keys: &[u32], nkeys: usize) -> (Vec<u64>, Vec<u32>) {
    let mut off = vec![0u64; nkeys + 1];
    for &k in keys {
        if k != u32::MAX {
            off[k as usize + 1] += 1;
        }
    }
    for i in 0..nkeys {
        off[i + 1] += off[i];
    }
    let mut fill = off.clone();
    let mut vals = vec![0u32; off[nkeys] as usize];
    for (v, &k) in keys.iter().enumerate() {
        if k != u32::MAX {
            vals[fill[k as usize] as usize] = v as u32;
            fill[k as usize] += 1;
        }
    }
    (off, vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(words: &[&str]) -> PoolBuilder {
        let mut p = PoolBuilder::new(true);
        for w in words {
            p.push(w.as_bytes()).unwrap();
        }
        p
    }

    #[test]
    fn modes() {
        let b = pool(&["foo", "foobar", "barfoo", "", "fo", "foo"]);
        assert_eq!(b.len(), 5);
        let p = Pool { blob: &b.blob, off: &b.off };
        assert_eq!(p.get(1), b"foobar");
        assert_eq!(p.get(3), b"");
        assert_eq!(p.matching(b"foo", Mode::Substring), vec![0, 1, 2]);
        assert_eq!(p.matching(b"foo", Mode::Prefix), vec![0, 1]);
        assert_eq!(p.matching(b"foo", Mode::Equal), vec![0]);
        assert_eq!(p.matching(b"", Mode::Equal), vec![3]);
        assert_eq!(p.matching(b"o", Mode::Substring), vec![0, 1, 2, 4]);
        assert_eq!(p.matching(b"zz", Mode::Substring), Vec::<u32>::new());
    }

    #[test]
    fn bare_mode() {
        let mut b = PoolBuilder::new(false);
        for w in ["foo", "foo()", "foo(x)", "foobar", "foo)(", "xfoo()", "fo()"] {
            b.push(w.as_bytes()).unwrap();
        }
        let p = Pool { blob: &b.blob, off: &b.off };
        assert_eq!(p.matching(b"foo", Mode::Bare), vec![0, 1, 4]);
        assert_eq!(p.matching(b"fo", Mode::Bare), vec![6]);
    }

    #[test]
    fn equal_does_not_skip_neighbours() {
        let mut b = PoolBuilder::new(false);
        for w in ["x", "x", "xx", "x"] {
            b.push(w.as_bytes()).unwrap();
        }
        let p = Pool { blob: &b.blob, off: &b.off };
        assert_eq!(p.matching(b"x", Mode::Equal), vec![0, 1, 3]);
        assert_eq!(p.matching(b"x", Mode::Prefix), vec![0, 1, 2, 3]);
    }

    #[test]
    fn front_coding_round_trips() {
        let mut b = PoolBuilder::new(false);
        let words: Vec<String> = (0..70).map(|i| format!("src_module_{}_fn_{}", i / 7, "x".repeat(i % 5))).collect();
        for w in &words {
            b.push(w.as_bytes()).unwrap();
        }
        b.push("кириллица_узел".as_bytes()).unwrap();
        b.push(b"").unwrap();
        let (blob, off) = front_code(&b);
        let p = FcPool { blob: &blob, off: &off };
        for (j, w) in words.iter().enumerate() {
            assert_eq!(p.get(j), w.as_bytes());
        }
        assert_eq!(p.get(70), "кириллица_узел".as_bytes());
        assert_eq!(p.get(71), b"");
        assert!(blob.len() < b.blob.len() / 2);
    }

    #[test]
    fn rejects_nul() {
        assert!(PoolBuilder::new(false).push(b"a\0b").is_err());
    }
}
