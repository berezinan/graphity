//! Tasks 1.3 / 1.4 and the D7 replace-over follow-up: what a memmap2-mapped
//! file allows on Windows.
//!
//! cargo run --release -p snap-core --example winfs_probe -- <command> ...
//!
//!   table    <dir>                          1.3 / 1.4 tables
//!   race     <dir> <std|movefileex> <iters> <readers> <size_kib>
//!                                           writer replaces graph.snap while
//!                                           readers open + map + verify
//!   diskfree <dir> <std|movefileex> <size_mib>
//!                                           is a replaced file's space freed
//!                                           when its last mapping closes
//!   hold     <path>                         map, print "ready", wait for a
//!                                           stdin line, verify, print result
//!
//! Replace modes: `std` is `std::fs::rename`; `movefileex` calls
//! `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` directly, which is what CPython's
//! `os.replace` does.

use memmap2::Mmap;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const LEN: usize = 64 * 1024;

fn payload(tag: u8) -> Vec<u8> {
    (0..LEN).map(|i| (i as u8).wrapping_mul(31).wrapping_add(tag)).collect()
}

fn write_file(p: &Path, tag: u8) {
    let mut f = File::create(p).unwrap();
    f.write_all(&payload(tag)).unwrap();
    f.sync_all().unwrap();
}

fn code(e: &io::Error) -> String {
    e.raw_os_error().map(|c| c.to_string()).unwrap_or_else(|| e.to_string())
}

fn res<T>(r: io::Result<T>) -> String {
    match r {
        Ok(_) => "ok".into(),
        Err(e) => format!("ERR {}", code(&e)),
    }
}

fn wide(p: &Path) -> Vec<u16> {
    OsStr::new(p).encode_wide().chain(std::iter::once(0)).collect()
}

fn move_file_ex(src: &Path, dst: &Path) -> io::Result<()> {
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_REPLACE_EXISTING};
    let (s, d) = (wide(src), wide(dst));
    let ok = unsafe { MoveFileExW(s.as_ptr(), d.as_ptr(), MOVEFILE_REPLACE_EXISTING) };
    if ok == 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

fn replace(mode: &str, src: &Path, dst: &Path) -> io::Result<()> {
    match mode {
        "std" => fs::rename(src, dst),
        "movefileex" => move_file_ex(src, dst),
        m => panic!("unknown replace mode {m}"),
    }
}

fn free_bytes(dir: &Path) -> u64 {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let d = wide(dir);
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    let ok = unsafe { GetDiskFreeSpaceExW(d.as_ptr(), &mut avail, &mut total, &mut free) };
    assert!(ok != 0, "GetDiskFreeSpaceExW: {}", io::Error::last_os_error());
    free
}

// ---- generation files: [tag u32 LE][body], body derived from tag ---------

fn gen_bytes(tag: u32, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    v.extend_from_slice(&tag.to_le_bytes());
    let salt = (tag as u8).wrapping_mul(7) ^ (tag >> 8) as u8;
    v.extend((4..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt)));
    v
}

fn write_gen(p: &Path, tag: u32, len: usize) -> io::Result<()> {
    let mut f = File::create(p)?;
    f.write_all(&gen_bytes(tag, len))?;
    f.sync_all()
}

/// Ok(tag) if the bytes are a complete, self-consistent generation.
fn verify_gen(m: &[u8], len: usize) -> Result<u32, &'static str> {
    if m.len() != len {
        return Err("wrong length");
    }
    let tag = u32::from_le_bytes(m[..4].try_into().unwrap());
    if m != &gen_bytes(tag, len)[..] {
        return Err("torn");
    }
    Ok(tag)
}

// ---- table -----------------------------------------------------------------

fn map(p: &Path, keep_handle: bool) -> (Mmap, Option<File>) {
    let f = File::open(p).unwrap();
    let m = unsafe { Mmap::map(&f).unwrap() };
    if keep_handle { (m, Some(f)) } else { drop(f); (m, None) }
}

fn intact(m: &Mmap, tag: u8) -> &'static str {
    if m[..] == payload(tag)[..] { "intact" } else { "CHANGED" }
}

fn case(dir: &Path, name: &str, keep_handle: bool, op: &dyn Fn(&Path, &Path) -> String) {
    let p = dir.join(format!("case.{}.snap", keep_handle as u8));
    let other = dir.join(format!("case.{}.other", keep_handle as u8));
    let _ = fs::remove_file(&p);
    let _ = fs::remove_file(&other);
    write_file(&p, 1);
    let (m, h) = map(&p, keep_handle);
    let r = op(&p, &other);
    let after_name = if p.exists() { "name present" } else { "name gone" };
    println!(
        "| {name} | {} | {r} | {} | {after_name} |",
        if keep_handle { "File + Mmap" } else { "Mmap only" },
        intact(&m, 1)
    );
    drop(h);
    drop(m);
    let _ = fs::remove_file(&p);
    let _ = fs::remove_file(&other);
}

fn table(dir: &Path) {
    println!("## 1.3 memmap2 over std::fs::File::open\n");
    println!("| operation | reader holds | result | reader view | name after |");
    println!("|---|---|---|---|---|");
    for keep in [true, false] {
        case(dir, "rename to new name", keep, &|p, o| res(fs::rename(p, o)));
        case(dir, "delete", keep, &|p, _| res(fs::remove_file(p)));
        for mode in ["std", "movefileex"] {
            case(dir, &format!("replace over ({mode})"), keep, &|p, o| {
                write_file(o, 2);
                res(replace(mode, o, p))
            });
        }
        case(dir, "overwrite in place (write at 0)", keep, &|p, _| {
            match OpenOptions::new().write(true).open(p) {
                Ok(f) => res(f.seek_write(&payload(3)[..16], 0)),
                Err(e) => format!("open: {}", res::<()>(Err(e))),
            }
        });
        case(dir, "truncate below mapped size", keep, &|p, _| {
            match OpenOptions::new().write(true).open(p) {
                Ok(f) => res(f.set_len(16)),
                Err(e) => format!("open: {}", res::<()>(Err(e))),
            }
        });
        case(dir, "append at end", keep, &|p, _| {
            match OpenOptions::new().append(true).open(p) {
                Ok(mut f) => res(f.write_all(b"tail")),
                Err(e) => format!("open: {}", res::<()>(Err(e))),
            }
        });
        case(dir, "delete, then create same name", keep, &|p, _| {
            let d = res(fs::remove_file(p));
            let c = res(File::create(p).map(|_| ()));
            format!("delete {d}; create {c}")
        });
    }

    println!("\n## 1.4 append-only handle (FILE_APPEND_DATA without FILE_WRITE_DATA)\n");
    println!("| attempt | result | file effect |");
    println!("|---|---|---|");
    let p = dir.join("append_only.bin");
    let attempts: [(&str, &dyn Fn(&File) -> io::Result<()>); 3] = [
        ("WriteFile with offset 0", &|f| f.seek_write(b"XXXX", 0).map(|_| ())),
        ("WriteFile with offset LEN/2", &|f| f.seek_write(b"XXXX", (LEN / 2) as u64).map(|_| ())),
        ("SetEndOfFile to 16 B", &|f| f.set_len(16)),
    ];
    for (label, op) in attempts {
        let _ = fs::remove_file(&p);
        write_file(&p, 1);
        let f = OpenOptions::new().append(true).open(&p).unwrap();
        let r = res(op(&f));
        drop(f);
        let now = fs::read(&p).unwrap();
        let orig = payload(1);
        let effect = if now.len() < LEN {
            format!("truncated to {} B", now.len())
        } else if now[..LEN] != orig[..] {
            "bytes below EOF CHANGED".to_string()
        } else if now.len() > LEN {
            format!("appended {} B at end, prefix intact", now.len() - LEN)
        } else {
            "unchanged".to_string()
        };
        println!("| {label} | {r} | {effect} |");
    }
    let _ = fs::remove_file(&p);
}

// ---- race ------------------------------------------------------------------

#[derive(Default, Debug)]
struct ReaderStats {
    ok: u64,
    open_err: BTreeMap<String, u64>,
    map_err: BTreeMap<String, u64>,
    bad: BTreeMap<&'static str, u64>,
    went_back: u64,
    tags_seen: u64,
}

fn race(dir: &Path, mode: &str, iters: u32, readers: usize, len: usize) {
    fs::create_dir_all(dir).unwrap();
    let live = dir.join("graph.snap");
    let tmp = dir.join("graph.snap.tmp");
    let _ = fs::remove_file(&tmp);
    write_gen(&live, 0, len).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = (0..readers)
        .map(|_| {
            let (stop, live) = (stop.clone(), live.clone());
            std::thread::spawn(move || {
                let mut s = ReaderStats::default();
                let mut last: Option<u32> = None;
                while !stop.load(Ordering::Relaxed) {
                    let f = match File::open(&live) {
                        Ok(f) => f,
                        Err(e) => { *s.open_err.entry(code(&e)).or_default() += 1; continue; }
                    };
                    let m = match unsafe { Mmap::map(&f) } {
                        Ok(m) => m,
                        Err(e) => { *s.map_err.entry(code(&e)).or_default() += 1; continue; }
                    };
                    drop(f);
                    match verify_gen(&m, len) {
                        Ok(tag) => {
                            s.ok += 1;
                            if let Some(l) = last {
                                if tag < l { s.went_back += 1; }
                                if tag != l { s.tags_seen += 1; }
                            }
                            last = Some(tag);
                        }
                        Err(why) => *s.bad.entry(why).or_default() += 1,
                    }
                }
                s
            })
        })
        .collect();

    let mut writer_err: BTreeMap<String, u64> = BTreeMap::new();
    let mut gave_up = 0u32;
    let t0 = Instant::now();
    for tag in 1..=iters {
        write_gen(&tmp, tag, len).unwrap();
        let mut done = false;
        for _ in 0..50 {
            match replace(mode, &tmp, &live) {
                Ok(()) => { done = true; break; }
                Err(e) => {
                    *writer_err.entry(code(&e)).or_default() += 1;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        if !done { gave_up += 1; }
    }
    let elapsed = t0.elapsed();
    stop.store(true, Ordering::Relaxed);
    let stats: Vec<ReaderStats> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let mut total = ReaderStats::default();
    for s in &stats {
        total.ok += s.ok;
        total.went_back += s.went_back;
        total.tags_seen += s.tags_seen;
        for (k, v) in &s.open_err { *total.open_err.entry(k.clone()).or_default() += v; }
        for (k, v) in &s.map_err { *total.map_err.entry(k.clone()).or_default() += v; }
        for (k, v) in &s.bad { *total.bad.entry(k).or_default() += v; }
    }
    let final_tag = verify_gen(&fs::read(&live).unwrap(), len);
    let leftovers: Vec<String> = fs::read_dir(dir).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    println!(
        "| {mode} | {iters} | {readers} | {} KiB | {:.1} s | writer retries {:?}, gave up {gave_up} | \
         reads ok {}, open err {:?}, map err {:?}, bad {:?}, went back {}, tag changes seen {} | \
         final {:?} | dir {:?} |",
        len / 1024, elapsed.as_secs_f64(), writer_err,
        total.ok, total.open_err, total.map_err, total.bad, total.went_back, total.tags_seen,
        final_tag, leftovers
    );
}

// ---- diskfree --------------------------------------------------------------

fn diskfree(dir: &Path, mode: &str, mib: usize) {
    fs::create_dir_all(dir).unwrap();
    let live = dir.join("graph.snap");
    let tmp = dir.join("graph.snap.tmp");
    let len = mib << 20;
    write_gen(&live, 1, len).unwrap();
    let f = File::open(&live).unwrap();
    let m = unsafe { Mmap::map(&f).unwrap() };
    drop(f);
    let f0 = free_bytes(dir);
    write_gen(&tmp, 2, 4096).unwrap();
    let r = res(replace(mode, &tmp, &live));
    std::thread::sleep(Duration::from_millis(300));
    let f1 = free_bytes(dir);
    let still = verify_gen(&m, len).map(|t| t == 1).unwrap_or(false);
    drop(m);
    std::thread::sleep(Duration::from_millis(1000));
    let f2 = free_bytes(dir);
    let names: Vec<String> = fs::read_dir(dir).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let mb = |a: u64, b: u64| (a as f64 - b as f64) / (1u64 << 20) as f64;
    println!(
        "| {mode} | {mib} MiB | replace {r} | old mapping intact {still} | \
         freed while mapped {:+.0} MiB | freed after unmap {:+.0} MiB | dir {:?} |",
        mb(f1, f0), mb(f2, f1), names
    );
    let _ = fs::remove_file(&live);
}

// ---- hold ------------------------------------------------------------------

fn hold(p: &Path) {
    let f = File::open(p).unwrap();
    let m = unsafe { Mmap::map(&f).unwrap() };
    drop(f);
    let before = m[..].to_vec();
    println!("ready");
    io::stdout().flush().unwrap();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).unwrap();
    println!("{}", if m[..] == before[..] { "intact" } else { "CHANGED" });
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| a.get(i).unwrap_or_else(|| panic!("missing argument {i}")).as_str();
    match arg(0) {
        "table" => table(&PathBuf::from(arg(1))),
        "race" => race(
            &PathBuf::from(arg(1)), arg(2),
            arg(3).parse().unwrap(), arg(4).parse().unwrap(),
            arg(5).parse::<usize>().unwrap() * 1024,
        ),
        "diskfree" => diskfree(&PathBuf::from(arg(1)), arg(2), arg(3).parse().unwrap()),
        "hold" => hold(&PathBuf::from(arg(1))),
        c => panic!("unknown command {c}"),
    }
}
