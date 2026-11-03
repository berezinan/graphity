//! Snapshot file writer and publication (D7).
//!
//! Sections are streamed into a temporary file, each starting on a 4 KiB
//! boundary. `finish` writes the header page at offset 0, appends the commit
//! footer and flushes to disk. `publish` replaces the live name over with
//! `std::fs::rename`, which works under live readers; `MoveFileExW` /
//! CPython `os.replace` does not (see README §1.3).

use crate::header::{
    encode_footer, fnv1a64, Header, HeaderMeta, Section, SectionKind, SnapError, FORMAT_VERSION,
    HEADER_LEN, PAGE,
};
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

fn io(e: std::io::Error) -> SnapError {
    SnapError::Io(e.to_string())
}

pub struct SnapWriter {
    path: PathBuf,
    out: BufWriter<File>,
    pos: u64,
    sections: Vec<Section>,
    open: Option<(SectionKind, u64)>,
}

impl SnapWriter {
    pub fn create(path: &Path) -> Result<Self, SnapError> {
        let f = File::create(path).map_err(io)?;
        let mut out = BufWriter::with_capacity(1 << 20, f);
        out.write_all(&vec![0u8; HEADER_LEN]).map_err(io)?;
        Ok(SnapWriter { path: path.to_owned(), out, pos: HEADER_LEN as u64, sections: vec![], open: None })
    }

    fn pad_to_page(&mut self) -> Result<(), SnapError> {
        let pad = (PAGE - self.pos % PAGE) % PAGE;
        if pad > 0 {
            self.out.write_all(&vec![0u8; pad as usize]).map_err(io)?;
            self.pos += pad;
        }
        Ok(())
    }

    pub fn begin_section(&mut self, kind: SectionKind) -> Result<(), SnapError> {
        if self.open.is_some() {
            return Err(SnapError::Corrupt("begin_section while a section is open".into()));
        }
        if self.sections.iter().any(|s| s.kind == kind) {
            return Err(SnapError::Corrupt(format!("section {kind:?} written twice")));
        }
        self.pad_to_page()?;
        self.open = Some((kind, self.pos));
        Ok(())
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<(), SnapError> {
        if self.open.is_none() {
            return Err(SnapError::Corrupt("write outside a section".into()));
        }
        self.out.write_all(bytes).map_err(io)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    pub fn end_section(&mut self) -> Result<(), SnapError> {
        let (kind, start) = self.open.take().ok_or_else(|| SnapError::Corrupt("no open section".into()))?;
        self.sections.push(Section { kind, offset: start, len: self.pos - start });
        Ok(())
    }

    pub fn section(&mut self, kind: SectionKind, bytes: &[u8]) -> Result<(), SnapError> {
        self.begin_section(kind)?;
        self.write(bytes)?;
        self.end_section()
    }

    /// Write header and footer, flush to disk. The file is complete after this.
    pub fn finish(mut self, meta: HeaderMeta) -> Result<PathBuf, SnapError> {
        if self.open.is_some() {
            return Err(SnapError::Corrupt("finish with an open section".into()));
        }
        let header = Header { format_version: FORMAT_VERSION, meta, sections: self.sections.clone() };
        let (page, used) = header.encode()?;
        let footer = encode_footer(self.pos + crate::header::FOOTER_LEN as u64, fnv1a64(&page[..used]), header.meta.generation);
        self.out.write_all(&footer).map_err(io)?;
        let mut f = self.out.into_inner().map_err(|e| io(e.into_error()))?;
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.write_all(&page).map_err(io)?;
        f.sync_all().map_err(io)?;
        Ok(self.path)
    }
}

/// Atomically replace `live` with the finished file at `tmp` (D7).
pub fn publish(tmp: &Path, live: &Path) -> Result<(), SnapError> {
    fs::rename(tmp, live).map_err(io)
}
