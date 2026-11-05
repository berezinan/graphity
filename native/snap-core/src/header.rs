//! Header page, section table and commit footer (task 2.1, design D3/D5/D7/D9).
//!
//! ```text
//! [header page, 4 KiB][section 1, 4 KiB aligned]...[section n][footer, 32 B]
//! ```
//!
//! The header page is written last (the writer seeks back to offset 0), and
//! the footer after it. A reader accepts a file only if the footer is present,
//! records the actual file length and carries the checksum of the header.

use std::fmt;

pub const MAGIC: [u8; 8] = *b"GRAPHSNP";
pub const FOOTER_MAGIC: [u8; 8] = *b"SNPCOMIT";
pub const FORMAT_VERSION: u32 = 1;
pub const PAGE: u64 = 4096;
pub const HEADER_LEN: usize = PAGE as usize;
pub const FOOTER_LEN: usize = 32;
const MAX_DICT_FILE: usize = 512;
const SECTION_ENTRY: usize = 24;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SectionKind {
    Dict = 1,
    Nodes = 2,
    Edges = 3,
    Adj = 4,
    Index = 5,
    Stats = 6,
    Cold = 7,
}

impl SectionKind {
    pub fn from_u32(v: u32) -> Option<Self> {
        use SectionKind::*;
        Some(match v {
            1 => Dict,
            2 => Nodes,
            3 => Edges,
            4 => Adj,
            5 => Index,
            6 => Stats,
            7 => Cold,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    pub offset: u64,
    pub len: u64,
}

/// What the `graph.json` the snapshot was built from looked like.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceInfo {
    pub size: u64,
    pub mtime_ns: i64,
    pub sha256: [u8; 32],
}

/// Where the string dictionary lives and how much of it this generation sees
/// (D8). An empty `file` means the DICT section of this same file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DictRef {
    pub file: String,
    pub watermark: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeaderMeta {
    pub generation: u64,
    pub norm_fingerprint: [u8; 32],
    pub source: SourceInfo,
    pub partition_hash: [u8; 32],
    pub node_count: u64,
    pub edge_count: u64,
    pub dict_ref: DictRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub format_version: u32,
    pub meta: HeaderMeta,
    pub sections: Vec<Section>,
}

/// Why a file was refused. Every variant tells the user to rebuild.
#[derive(Debug, PartialEq, Eq)]
pub enum SnapError {
    Io(String),
    NotASnapshot,
    VersionMismatch { found: u32, expected: u32 },
    Incomplete(&'static str),
    Corrupt(String),
    FingerprintMismatch { found: String, expected: String },
}

impl fmt::Display for SnapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hint = "rebuild the snapshot from graph.json";
        match self {
            SnapError::Io(e) => write!(f, "snapshot cannot be read: {e}"),
            SnapError::NotASnapshot => write!(f, "not a graph snapshot (bad magic); {hint}"),
            SnapError::VersionMismatch { found, expected } => write!(
                f,
                "snapshot format version {found} is not supported (this reader needs {expected}); {hint}"
            ),
            SnapError::Incomplete(why) => {
                write!(f, "snapshot is incomplete, the write did not finish ({why}); {hint}")
            }
            SnapError::Corrupt(why) => write!(f, "snapshot is corrupt ({why}); {hint}"),
            SnapError::FingerprintMismatch { found, expected } => write!(
                f,
                "snapshot was built with different node normalisation \
                 (fingerprint {found}, current code {expected}); {hint}"
            ),
        }
    }
}

impl std::error::Error for SnapError {}

/// FNV-1a 64: header checksum only, not a security property.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapError> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len());
        let end = end.ok_or_else(|| SnapError::Corrupt("header overruns its page".into()))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, SnapError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, SnapError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, SnapError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, SnapError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn h32(&mut self) -> Result<[u8; 32], SnapError> {
        Ok(self.take(32)?.try_into().unwrap())
    }
}

impl Header {
    /// Encode into a full header page. Returns the page and the number of
    /// meaningful bytes, which is what the footer checksum covers.
    pub fn encode(&self) -> Result<(Vec<u8>, usize), SnapError> {
        let m = &self.meta;
        if m.dict_ref.file.len() > MAX_DICT_FILE {
            return Err(SnapError::Corrupt("dict_ref.file too long".into()));
        }
        let mut v = Vec::with_capacity(HEADER_LEN);
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&self.format_version.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes()); // used length, patched below
        v.extend_from_slice(&m.generation.to_le_bytes());
        v.extend_from_slice(&m.norm_fingerprint);
        v.extend_from_slice(&m.source.size.to_le_bytes());
        v.extend_from_slice(&m.source.mtime_ns.to_le_bytes());
        v.extend_from_slice(&m.source.sha256);
        v.extend_from_slice(&m.partition_hash);
        v.extend_from_slice(&m.node_count.to_le_bytes());
        v.extend_from_slice(&m.edge_count.to_le_bytes());
        v.extend_from_slice(&m.dict_ref.watermark.to_le_bytes());
        v.extend_from_slice(&(m.dict_ref.file.len() as u16).to_le_bytes());
        v.extend_from_slice(m.dict_ref.file.as_bytes());
        v.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for s in &self.sections {
            v.extend_from_slice(&(s.kind as u32).to_le_bytes());
            v.extend_from_slice(&0u32.to_le_bytes());
            v.extend_from_slice(&s.offset.to_le_bytes());
            v.extend_from_slice(&s.len.to_le_bytes());
        }
        let used = v.len();
        if used > HEADER_LEN {
            return Err(SnapError::Corrupt("header does not fit its page".into()));
        }
        v[12..16].copy_from_slice(&(used as u32).to_le_bytes());
        v.resize(HEADER_LEN, 0);
        Ok((v, used))
    }

    /// Decode and validate everything that does not need the expected
    /// fingerprint: magic, version, footer, checksum, section bounds.
    pub fn decode(file: &[u8]) -> Result<Header, SnapError> {
        // The writer fills the header page last, so a write that died before
        // finish() leaves zeros where the magic goes.
        if file.len() < 12 || file[..8].iter().all(|&b| b == 0) {
            return Err(SnapError::Incomplete("header not written"));
        }
        if file[..8] != MAGIC {
            return Err(SnapError::NotASnapshot);
        }
        let version = u32::from_le_bytes(file[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(SnapError::VersionMismatch { found: version, expected: FORMAT_VERSION });
        }
        if file.len() < HEADER_LEN + FOOTER_LEN {
            return Err(SnapError::Incomplete("file shorter than header and footer"));
        }
        let footer = &file[file.len() - FOOTER_LEN..];
        if footer[..8] != FOOTER_MAGIC {
            return Err(SnapError::Incomplete("commit marker missing"));
        }
        let rec_len = u64::from_le_bytes(footer[8..16].try_into().unwrap());
        if rec_len != file.len() as u64 {
            return Err(SnapError::Incomplete("recorded length differs from file length"));
        }
        let checksum = u64::from_le_bytes(footer[16..24].try_into().unwrap());
        let footer_gen = u64::from_le_bytes(footer[24..32].try_into().unwrap());

        let page = &file[..HEADER_LEN];
        let used = u32::from_le_bytes(page[12..16].try_into().unwrap()) as usize;
        if used < 16 || used > HEADER_LEN {
            return Err(SnapError::Corrupt("bad header length".into()));
        }
        if fnv1a64(&page[..used]) != checksum {
            return Err(SnapError::Corrupt("header checksum mismatch".into()));
        }

        let mut c = Cursor { b: &page[..used], at: 16 };
        let generation = c.u64()?;
        let norm_fingerprint = c.h32()?;
        let source = SourceInfo { size: c.u64()?, mtime_ns: c.i64()?, sha256: c.h32()? };
        let partition_hash = c.h32()?;
        let node_count = c.u64()?;
        let edge_count = c.u64()?;
        let watermark = c.u64()?;
        let flen = c.u16()? as usize;
        if flen > MAX_DICT_FILE {
            return Err(SnapError::Corrupt("dict_ref.file too long".into()));
        }
        let file_name = std::str::from_utf8(c.take(flen)?)
            .map_err(|_| SnapError::Corrupt("dict_ref.file is not UTF-8".into()))?
            .to_owned();
        let n = c.u32()? as usize;
        if n.checked_mul(SECTION_ENTRY).map_or(true, |b| c.at + b > used) {
            return Err(SnapError::Corrupt("section table overruns header".into()));
        }
        let body_end = (file.len() - FOOTER_LEN) as u64;
        let mut sections = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = c.u32()?;
            let kind = SectionKind::from_u32(raw)
                .ok_or_else(|| SnapError::Corrupt(format!("unknown section kind {raw}")))?;
            let _reserved = c.u32()?;
            let offset = c.u64()?;
            let len = c.u64()?;
            let fits = offset >= PAGE && offset % PAGE == 0 && offset.checked_add(len).map_or(false, |e| e <= body_end);
            if !fits {
                return Err(SnapError::Corrupt(format!("section {kind:?} out of bounds")));
            }
            if sections.iter().any(|s: &Section| s.kind == kind) {
                return Err(SnapError::Corrupt(format!("section {kind:?} repeated")));
            }
            sections.push(Section { kind, offset, len });
        }
        if footer_gen != generation {
            return Err(SnapError::Corrupt("footer generation differs from header".into()));
        }
        Ok(Header {
            format_version: version,
            meta: HeaderMeta {
                generation,
                norm_fingerprint,
                source,
                partition_hash,
                node_count,
                edge_count,
                dict_ref: DictRef { file: file_name, watermark },
            },
            sections,
        })
    }

    pub fn check_fingerprint(&self, expected: &[u8; 32]) -> Result<(), SnapError> {
        if &self.meta.norm_fingerprint != expected {
            return Err(SnapError::FingerprintMismatch {
                found: hex(&self.meta.norm_fingerprint),
                expected: hex(expected),
            });
        }
        Ok(())
    }

    pub fn section(&self, kind: SectionKind) -> Option<Section> {
        self.sections.iter().copied().find(|s| s.kind == kind)
    }
}

pub fn encode_footer(file_len: u64, header_checksum: u64, generation: u64) -> [u8; FOOTER_LEN] {
    let mut f = [0u8; FOOTER_LEN];
    f[..8].copy_from_slice(&FOOTER_MAGIC);
    f[8..16].copy_from_slice(&file_len.to_le_bytes());
    f[16..24].copy_from_slice(&header_checksum.to_le_bytes());
    f[24..32].copy_from_slice(&generation.to_le_bytes());
    f
}
