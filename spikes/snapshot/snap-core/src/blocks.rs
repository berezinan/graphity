//! Named blocks inside one section, each 8-byte aligned.
//!
//! ```text
//! [u32 count][u32 0][count x {name [u8; 32], offset u64, len u64}][blocks ...]
//! ```
//!
//! Offsets are relative to the section start; sections start on a page, so
//! every block is 8-byte aligned in the mapping. Arrays are little-endian.

use crate::header::SnapError;
use crate::writer::SnapWriter;
use std::collections::HashMap;

const NAME: usize = 32;
const ENTRY: usize = NAME + 16;

pub enum Data {
    Bytes(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl Data {
    fn bytes(&self) -> &[u8] {
        match self {
            Data::Bytes(v) => v,
            Data::U32(v) => as_bytes(v),
            Data::U64(v) => as_bytes(v),
        }
    }
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices (u32 / u64) reinterpreted as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn align8(x: usize) -> usize {
    (x + 7) & !7
}

#[derive(Default)]
pub struct BlocksBuilder {
    items: Vec<(String, Data)>,
}

impl BlocksBuilder {
    pub fn put(&mut self, name: impl Into<String>, d: Data) {
        self.items.push((name.into(), d));
    }

    pub fn write(&self, w: &mut SnapWriter) -> Result<(), SnapError> {
        let head = 8 + ENTRY * self.items.len();
        let mut dir = Vec::with_capacity(head);
        dir.extend_from_slice(&(self.items.len() as u32).to_le_bytes());
        dir.extend_from_slice(&0u32.to_le_bytes());
        let mut pos = align8(head);
        let mut offs = Vec::with_capacity(self.items.len());
        for (name, d) in &self.items {
            if name.len() > NAME {
                return Err(SnapError::Corrupt(format!("block name too long: {name}")));
            }
            let mut nm = [0u8; NAME];
            nm[..name.len()].copy_from_slice(name.as_bytes());
            dir.extend_from_slice(&nm);
            dir.extend_from_slice(&(pos as u64).to_le_bytes());
            dir.extend_from_slice(&(d.bytes().len() as u64).to_le_bytes());
            offs.push(pos);
            pos = align8(pos + d.bytes().len());
        }
        w.write(&dir)?;
        let mut at = head;
        for ((_, d), &o) in self.items.iter().zip(&offs) {
            if o > at {
                w.write(&vec![0u8; o - at])?;
            }
            w.write(d.bytes())?;
            at = o + d.bytes().len();
        }
        Ok(())
    }
}

/// Absolute byte ranges `(offset, len)` of every block in a section.
pub fn parse(
    file: &[u8],
    off: usize,
    len: usize,
    map: &mut HashMap<String, (usize, usize)>,
) -> Result<(), SnapError> {
    let bad = |why: &str| SnapError::Corrupt(format!("block directory: {why}"));
    let sec = &file[off..off + len];
    if sec.len() < 8 {
        return Err(bad("too short"));
    }
    let n = u32::from_le_bytes(sec[..4].try_into().unwrap()) as usize;
    if n.checked_mul(ENTRY).map_or(true, |b| 8 + b > sec.len()) {
        return Err(bad("overruns section"));
    }
    for k in 0..n {
        let e = &sec[8 + k * ENTRY..8 + (k + 1) * ENTRY];
        let end = e[..NAME].iter().position(|&b| b == 0).unwrap_or(NAME);
        let name = std::str::from_utf8(&e[..end]).map_err(|_| bad("name not UTF-8"))?;
        let o = u64::from_le_bytes(e[NAME..NAME + 8].try_into().unwrap()) as usize;
        let l = u64::from_le_bytes(e[NAME + 8..].try_into().unwrap()) as usize;
        if o % 8 != 0 || o.checked_add(l).map_or(true, |x| x > len) {
            return Err(bad(&format!("block {name} out of bounds")));
        }
        if map.insert(name.to_owned(), (off + o, l)).is_some() {
            return Err(bad(&format!("block {name} repeated")));
        }
    }
    Ok(())
}

/// View a block as `&[T]`. Blocks are 8-aligned in a page-aligned mapping.
pub fn cast<T: Copy>(b: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert!(b.len() % size == 0 && (b.as_ptr() as usize) % std::mem::align_of::<T>() == 0);
    // SAFETY: alignment and length checked above; T is plain old data.
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, b.len() / size) }
}
