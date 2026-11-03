//! Memory-mapped snapshot reader (D6). Opening validates the header page and
//! the footer only; section bytes are not touched until asked for.

use crate::header::{Header, SectionKind, SnapError};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct Snapshot {
    map: Mmap,
    pub header: Header,
}

impl Snapshot {
    /// Open `path` and check it against the caller's normalisation fingerprint.
    pub fn open(path: &Path, expected_fingerprint: &[u8; 32]) -> Result<Snapshot, SnapError> {
        let s = Self::open_unchecked(path)?;
        s.header.check_fingerprint(expected_fingerprint)?;
        Ok(s)
    }

    /// Open without the fingerprint check (inspection tools only).
    pub fn open_unchecked(path: &Path) -> Result<Snapshot, SnapError> {
        // std opens with FILE_SHARE_READ | WRITE | DELETE, so the writer can
        // still replace the name over (README §1.3).
        let f = File::open(path).map_err(|e| SnapError::Io(e.to_string()))?;
        let map = unsafe { Mmap::map(&f) }.map_err(|e| SnapError::Io(e.to_string()))?;
        drop(f);
        let header = Header::decode(&map)?;
        Ok(Snapshot { map, header })
    }

    pub fn section(&self, kind: SectionKind) -> Option<&[u8]> {
        let s = self.header.section(kind)?;
        Some(&self.map[s.offset as usize..(s.offset + s.len) as usize])
    }

    pub fn file_len(&self) -> usize {
        self.map.len()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}
