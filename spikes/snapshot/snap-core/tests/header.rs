//! Task 2.1: the reader refuses a snapshot of another format version, with
//! another normalisation fingerprint, or whose write did not finish.

use snap_core::header::{SectionKind, SnapError, FOOTER_LEN, FORMAT_VERSION, HEADER_LEN, PAGE};
use snap_core::{publish, DictRef, HeaderMeta, SnapWriter, Snapshot, SourceInfo};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const FP: [u8; 32] = [7u8; 32];

fn meta() -> HeaderMeta {
    HeaderMeta {
        generation: 42,
        norm_fingerprint: FP,
        source: SourceInfo { size: 155_500_000, mtime_ns: 1_762_100_000_123_456_789, sha256: [9u8; 32] },
        partition_hash: [3u8; 32],
        node_count: 40_988,
        edge_count: 96_315,
        dict_ref: DictRef { file: String::new(), watermark: 12_345 },
    }
}

fn write_ok(dir: &Path) -> PathBuf {
    let p = dir.join("graph.snap");
    let mut w = SnapWriter::create(&p).unwrap();
    w.section(SectionKind::Dict, b"dictionary bytes").unwrap();
    w.begin_section(SectionKind::Nodes).unwrap();
    w.write(&[1u8; 5000]).unwrap();
    w.write(&[2u8; 10]).unwrap();
    w.end_section().unwrap();
    w.finish(meta()).unwrap();
    p
}

fn err(p: &Path, fp: &[u8; 32]) -> SnapError {
    match Snapshot::open(p, fp) {
        Ok(_) => panic!("snapshot was accepted"),
        Err(e) => e,
    }
}

#[test]
fn round_trip() {
    let d = tempfile::tempdir().unwrap();
    let p = write_ok(d.path());
    let s = Snapshot::open(&p, &FP).unwrap();
    assert_eq!(s.header.format_version, FORMAT_VERSION);
    assert_eq!(s.header.meta, meta());
    assert_eq!(s.section(SectionKind::Dict).unwrap(), b"dictionary bytes");
    let nodes = s.section(SectionKind::Nodes).unwrap();
    assert_eq!(nodes.len(), 5010);
    assert!(nodes[..5000].iter().all(|&b| b == 1) && nodes[5000..].iter().all(|&b| b == 2));
    assert!(s.section(SectionKind::Edges).is_none());
    for sec in &s.header.sections {
        assert_eq!(sec.offset % PAGE, 0, "{:?} not page aligned", sec.kind);
    }
}

#[test]
fn rejects_other_format_version() {
    let d = tempfile::tempdir().unwrap();
    let p = write_ok(d.path());
    let mut f = OpenOptions::new().write(true).open(&p).unwrap();
    f.seek(SeekFrom::Start(8)).unwrap();
    f.write_all(&(FORMAT_VERSION + 1).to_le_bytes()).unwrap();
    drop(f);
    let e = err(&p, &FP);
    assert_eq!(e, SnapError::VersionMismatch { found: FORMAT_VERSION + 1, expected: FORMAT_VERSION });
    assert!(e.to_string().contains("rebuild the snapshot"), "{e}");
}

#[test]
fn rejects_other_fingerprint() {
    let d = tempfile::tempdir().unwrap();
    let p = write_ok(d.path());
    let e = err(&p, &[8u8; 32]);
    assert!(matches!(e, SnapError::FingerprintMismatch { .. }), "{e:?}");
    let msg = e.to_string();
    assert!(msg.contains("normalisation") && msg.contains("rebuild the snapshot"), "{msg}");
}

#[test]
fn rejects_truncated_file_at_every_cut() {
    let d = tempfile::tempdir().unwrap();
    let p = write_ok(d.path());
    let full = fs::read(&p).unwrap();
    let cut = d.path().join("cut.snap");
    let cuts = [12, HEADER_LEN, HEADER_LEN + 100, full.len() - FOOTER_LEN, full.len() - 1];
    for n in cuts {
        fs::write(&cut, &full[..n]).unwrap();
        let e = err(&cut, &FP);
        assert!(matches!(e, SnapError::Incomplete(_)), "cut at {n}: {e:?}");
        assert!(e.to_string().contains("incomplete"), "{e}");
    }
}

#[test]
fn rejects_file_without_commit_marker() {
    // A writer that died after the sections and before finish(): header page
    // is still zeros, no footer.
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("graph.snap.123.tmp");
    let mut w = SnapWriter::create(&p).unwrap();
    w.section(SectionKind::Dict, &[5u8; 9000]).unwrap();
    drop(w);
    assert_eq!(err(&p, &FP), SnapError::Incomplete("header not written"));

    // Not a snapshot at all.
    fs::write(&p, b"{\"nodes\": [], \"links\": []}").unwrap();
    assert_eq!(err(&p, &FP), SnapError::NotASnapshot);

    // Header written but footer cut off.
    let ok = write_ok(d.path());
    let full = fs::read(&ok).unwrap();
    let mut no_footer = full[..full.len() - FOOTER_LEN].to_vec();
    no_footer.extend_from_slice(&[0u8; FOOTER_LEN]);
    fs::write(&p, &no_footer).unwrap();
    assert_eq!(err(&p, &FP), SnapError::Incomplete("commit marker missing"));
}

#[test]
fn rejects_tampered_header() {
    let d = tempfile::tempdir().unwrap();
    let p = write_ok(d.path());
    let mut f = OpenOptions::new().write(true).open(&p).unwrap();
    f.seek(SeekFrom::Start(16)).unwrap(); // generation
    f.write_all(&43u64.to_le_bytes()).unwrap();
    drop(f);
    assert!(matches!(err(&p, &FP), SnapError::Corrupt(_)));
}

#[test]
fn publish_replaces_over_a_mapped_generation() {
    let d = tempfile::tempdir().unwrap();
    let live = write_ok(d.path());
    let old = Snapshot::open(&live, &FP).unwrap();

    let tmp = d.path().join("graph.snap.1.tmp");
    let mut w = SnapWriter::create(&tmp).unwrap();
    w.section(SectionKind::Dict, b"second generation").unwrap();
    let mut m = meta();
    m.generation = 43;
    w.finish(m).unwrap();
    publish(&tmp, &live).unwrap();

    assert_eq!(old.header.meta.generation, 42);
    assert_eq!(old.section(SectionKind::Dict).unwrap(), b"dictionary bytes");
    let new = Snapshot::open(&live, &FP).unwrap();
    assert_eq!(new.header.meta.generation, 43);
    assert_eq!(new.section(SectionKind::Dict).unwrap(), b"second generation");
    assert!(!tmp.exists());
}
