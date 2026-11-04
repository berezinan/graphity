//! PyO3 surface of the snapshot: `Encoder` for the Python writer, `Snap` for
//! `SnapBackend`. Node and link indices are positions in graph.json.

use pyo3::exceptions::{PyBufferError, PyIOError, PyKeyError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::os::raw::{c_int, c_void};
use snap_core::graph::Graph;
use snap_core::header::hex;
use snap_core::pool::Mode;
use snap_core::{DictRef, Encoder, HeaderMeta, SnapError, Snapshot, SourceInfo};
use std::path::Path;
use std::sync::Arc;

fn err(e: SnapError) -> PyErr {
    match e {
        SnapError::Io(m) => PyIOError::new_err(m),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn h32(b: &[u8], what: &str) -> PyResult<[u8; 32]> {
    b.try_into().map_err(|_| PyValueError::new_err(format!("{what} must be 32 bytes")))
}

fn mode(s: &str) -> PyResult<Mode> {
    Mode::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown match mode {s}")))
}

fn text(b: &[u8]) -> PyResult<String> {
    String::from_utf8(b.to_vec()).map_err(|_| PyValueError::new_err("snapshot string is not UTF-8"))
}

#[pyclass(name = "Encoder", unsendable)]
struct PyEncoder {
    inner: Option<Encoder>,
}

impl PyEncoder {
    fn enc(&mut self) -> PyResult<&mut Encoder> {
        self.inner.as_mut().ok_or_else(|| PyValueError::new_err("encoder already finished"))
    }
}

#[pymethods]
impl PyEncoder {
    #[new]
    fn new(node_fields: Vec<String>, link_fields: Vec<String>, directed: bool) -> Self {
        PyEncoder { inner: Some(Encoder::new(node_fields, link_fields, directed)) }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_nodes(
        &mut self,
        ids: Vec<String>,
        idl: Vec<Option<String>>,
        idn: Vec<Option<String>>,
        search: Vec<Vec<String>>,
        vals: Vec<Vec<Option<String>>>,
        llen: Vec<u32>,
        cold: Vec<Option<String>>,
    ) -> PyResult<()> {
        self.enc()?.add_nodes(&ids, &idl, &idn, &search, &vals, &llen, &cold).map_err(err)
    }

    fn add_links(
        &mut self,
        src: Vec<u32>,
        tgt: Vec<u32>,
        vals: Vec<Vec<Option<String>>>,
        cold: Vec<Option<String>>,
    ) -> PyResult<()> {
        self.enc()?.add_links(&src, &tgt, &vals, &cold).map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        py: Python<'_>,
        path: &str,
        generation: u64,
        fingerprint: &[u8],
        source_size: u64,
        source_mtime_ns: i64,
        source_sha256: &[u8],
        partition_hash: &[u8],
        meta_json: &str,
        stats_json: &str,
        graph_json: &str,
        comm_off: Vec<u64>,
        comm_mem: Vec<u32>,
    ) -> PyResult<Vec<(String, f64)>> {
        let enc = self.inner.take().ok_or_else(|| PyValueError::new_err("encoder already finished"))?;
        let meta = HeaderMeta {
            generation,
            norm_fingerprint: h32(fingerprint, "fingerprint")?,
            source: SourceInfo { size: source_size, mtime_ns: source_mtime_ns, sha256: h32(source_sha256, "sha256")? },
            partition_hash: h32(partition_hash, "partition_hash")?,
            node_count: 0,
            edge_count: 0,
            dict_ref: DictRef { file: String::new(), watermark: 0 },
        };
        let (path, meta_json, stats_json, graph_json) =
            (path.to_owned(), meta_json.to_owned(), stats_json.to_owned(), graph_json.to_owned());
        let times = py
            .allow_threads(move || {
                enc.finish(Path::new(&path), meta, &meta_json, &stats_json, &graph_json, comm_off, comm_mem)
            })
            .map_err(err)?;
        Ok(times.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
    }
}

#[pyclass(frozen)]
struct Snap {
    g: Arc<Graph>,
}

/// Read-only buffer over one block of the mapping, without a copy. Holds the
/// generation alive (`Arc`) for as long as any memoryview of it exists.
#[pyclass(frozen)]
struct Block {
    g: Arc<Graph>,
    off: usize,
    len: usize,
}

#[pymethods]
impl Block {
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("snapshot blocks are read-only"));
        }
        let b = slf.get();
        let bytes = &b.g.snap.bytes()[b.off..b.off + b.len];
        (*view).obj = slf.clone().into_any().into_ptr();
        (*view).buf = bytes.as_ptr() as *mut c_void;
        (*view).len = bytes.len() as ffi::Py_ssize_t;
        (*view).readonly = 1;
        (*view).itemsize = 1;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            c"B".as_ptr() as *mut _
        } else {
            std::ptr::null_mut()
        };
        (*view).ndim = 1;
        (*view).shape =
            if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND { &mut (*view).len } else { std::ptr::null_mut() };
        (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            &mut (*view).itemsize
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

#[pymethods]
impl Snap {
    #[new]
    fn new(path: &str, fingerprint: &[u8]) -> PyResult<Self> {
        let g = Graph::open(Path::new(path), &h32(fingerprint, "fingerprint")?).map_err(err)?;
        Ok(Snap { g: Arc::new(g) })
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.g.n
    }

    #[getter]
    fn link_count(&self) -> usize {
        self.g.e
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.g.snap.header.meta.generation
    }

    #[getter]
    fn file_len(&self) -> usize {
        self.g.snap.file_len()
    }

    /// (size, mtime_ns, sha256 hex) of the graph.json the snapshot was built from.
    #[getter]
    fn source(&self) -> (u64, i64, String) {
        let s = &self.g.snap.header.meta.source;
        (s.size, s.mtime_ns, hex(&s.sha256))
    }

    #[getter]
    fn partition_hash(&self) -> String {
        hex(&self.g.snap.header.meta.partition_hash)
    }

    /// Section name -> byte length, for the size report.
    fn section_sizes(&self) -> Vec<(String, u64)> {
        self.g.snap.header.sections.iter().map(|s| (format!("{:?}", s.kind), s.len)).collect()
    }

    fn block_text(&self, name: &str) -> PyResult<Option<String>> {
        self.g.block(name).map(text).transpose()
    }

    fn block_bytes<'py>(&self, py: Python<'py>, name: &str) -> Option<Bound<'py, PyBytes>> {
        self.g.block(name).map(|b| PyBytes::new(py, b))
    }

    /// Zero-copy buffer over a block: `memoryview(snap.view("l.src")).cast("I")`.
    fn view(&self, name: &str) -> PyResult<Block> {
        let (off, len) = self.g.block_range(name).ok_or_else(|| PyKeyError::new_err(name.to_owned()))?;
        Ok(Block { g: self.g.clone(), off, len })
    }

    /// `_find_node_tiers` (no `::`, not a dotted Python symbol), see Graph::tiers.
    fn tiers(&self, py: Python<'_>, term: String, nq: String, level: u8) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let g = self.g.clone();
        let [a, b, c, d] = py.allow_threads(move || g.tiers(term.as_bytes(), nq.as_bytes(), level));
        (a, b, c, d)
    }

    /// Merged field codes and last-link endpoints per traversal pair (flat u32).
    fn pair_codes<'py>(&self, py: Python<'py>, us: Vec<u32>, vs: Vec<u32>, fields: Vec<String>) -> Bound<'py, PyBytes> {
        let g = self.g.clone();
        let out = py.allow_threads(move || g.pair_codes(&us, &vs, &fields));
        PyBytes::new(py, snap_core::blocks::as_bytes(&out))
    }

    #[pyo3(signature = (visited, seen_u, seen_v, filt=None))]
    fn complete_induced(
        &self,
        py: Python<'_>,
        visited: Vec<u32>,
        seen_u: Vec<u32>,
        seen_v: Vec<u32>,
        filt: Option<Vec<u32>>,
    ) -> (Vec<u32>, Vec<u32>) {
        let g = self.g.clone();
        py.allow_threads(move || g.complete_induced(&visited, &seen_u, &seen_v, filt.as_deref()))
    }

    fn cold_contains(&self, kind: &str, needle: &str) -> bool {
        self.g.cold_contains(kind, needle.as_bytes())
    }

    fn node_id(&self, i: usize) -> PyResult<String> {
        text(&self.g.id(i))
    }

    fn node_ids(&self, idx: Vec<u32>) -> PyResult<Vec<String>> {
        idx.iter().map(|&i| text(&self.g.id(i as usize))).collect()
    }

    /// Value codes of `fields` for node (`kind="n"`) or link (`"l"`) `i`, plus its cold JSON.
    fn record(&self, kind: &str, fields: Vec<String>, i: usize) -> PyResult<(Vec<u32>, String)> {
        let prefix = if kind == "n" { "n.v." } else { "l.v." };
        Ok((self.g.codes(prefix, &fields, i), text(self.g.cold(kind, i))?))
    }

    fn value(&self, kind: &str, field: &str, code: u32) -> PyResult<String> {
        let name = if kind == "n" { format!("nv.{field}") } else { format!("lv.{field}") };
        let p = self.g.pool(&name).ok_or_else(|| PyValueError::new_err(format!("no value pool {name}")))?;
        text(p.get(code as usize - 1))
    }

    fn pool_size(&self, kind: &str, field: &str) -> usize {
        let name = if kind == "n" { format!("nv.{field}") } else { format!("lv.{field}") };
        self.g.pool(&name).map_or(0, |p| p.len())
    }

    fn link_endpoints(&self, j: usize) -> (u32, u32) {
        (self.g.u32s("l.src")[j], self.g.u32s("l.tgt")[j])
    }

    fn match_nodes(&self, field: &str, mode_: &str, needles: Vec<String>) -> PyResult<Vec<u32>> {
        let m = mode(mode_)?;
        let nd: Vec<&[u8]> = needles.iter().map(|s| s.as_bytes()).collect();
        Ok(self.g.match_nodes(field, m, &nd))
    }

    fn match_ids(&self, kind: &str, mode_: &str, needles: Vec<String>) -> PyResult<Vec<u32>> {
        let m = mode(mode_)?;
        let nd: Vec<&[u8]> = needles.iter().map(|s| s.as_bytes()).collect();
        Ok(self.g.match_ids(kind, m, &nd))
    }

    fn df(&self, py: Python<'_>, terms: Vec<String>) -> Vec<u64> {
        let g = self.g.clone();
        py.allow_threads(move || {
            let t: Vec<&[u8]> = terms.iter().map(|s| s.as_bytes()).collect();
            g.df(&t)
        })
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn score<'py>(
        &self,
        py: Python<'py>,
        terms: Vec<String>,
        w: Vec<f64>,
        joined: String,
        jw: f64,
        cov: Vec<f64>,
        collect: bool,
    ) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<Option<(f64, u32)>>) {
        let g = self.g.clone();
        let out = py.allow_threads(move || {
            let t: Vec<&[u8]> = terms.iter().map(|s| s.as_bytes()).collect();
            g.score(&t, &w, joined.as_bytes(), jw, &cov, collect)
        });
        (
            PyBytes::new(py, snap_core::blocks::as_bytes(&out.idx)),
            PyBytes::new(py, snap_core::blocks::as_bytes(&out.score)),
            out.best,
        )
    }

    #[pyo3(signature = (i, filt=None))]
    fn tneighbors(&self, i: usize, filt: Option<Vec<u32>>) -> Vec<u32> {
        self.g.tneighbors(i, filt.as_deref())
    }

    #[pyo3(signature = (i, filt=None))]
    fn tdegree(&self, i: usize, filt: Option<Vec<u32>>) -> u32 {
        self.g.tdegree(i, filt.as_deref())
    }

    #[pyo3(signature = (filt=None))]
    fn hub_threshold(&self, py: Python<'_>, filt: Option<Vec<u32>>) -> u32 {
        let g = self.g.clone();
        py.allow_threads(move || g.hub_threshold(filt.as_deref()))
    }

    fn pair_links(&self, i: usize, j: usize) -> Vec<u32> {
        self.g.pair_links(i, j)
    }

    fn out_links(&self, i: usize) -> Vec<u32> {
        self.g.out_links(i).to_vec()
    }

    fn in_links(&self, i: usize) -> Vec<u32> {
        self.g.in_links(i).to_vec()
    }

    fn links_between(&self, i: usize, j: usize) -> Vec<u32> {
        self.g.links_between(i, j)
    }

    fn deg_dir(&self, i: usize) -> u32 {
        self.g.deg_dir(i)
    }

    fn deg_q(&self, i: usize) -> u32 {
        self.g.u32s("n.degq")[i]
    }

    fn label_len(&self, i: usize) -> u32 {
        self.g.u32s("n.llen")[i]
    }

    fn path_expand(&self, frontier: Vec<u32>, forward: bool, undirected: bool, contains: u32) -> Vec<Vec<u32>> {
        self.g.path_expand(&frontier, forward, undirected, contains)
    }

    /// Search string `field` of node `i` (for tests).
    fn search_string(&self, field: &str, i: usize) -> PyResult<String> {
        let f = snap_core::encode::SEARCH
            .iter()
            .position(|s| *s == field)
            .ok_or_else(|| PyValueError::new_err(format!("unknown search field {field}")))?;
        text(self.g.sget(f, i))
    }
}

/// Atomically replace `live` with the finished snapshot `tmp` (D7).
#[pyfunction]
fn publish(tmp: &str, live: &str) -> PyResult<()> {
    snap_core::publish(Path::new(tmp), Path::new(live)).map_err(err)
}

/// Generation of the snapshot at `path`, or None when absent or not readable.
#[pyfunction]
fn peek_generation(path: &str) -> Option<u64> {
    Snapshot::open_unchecked(Path::new(path)).ok().map(|s| s.header.meta.generation)
}

#[pyfunction]
fn format_version() -> u32 {
    snap_core::FORMAT_VERSION
}

#[pymodule]
fn graphify_snap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoder>()?;
    m.add_class::<Snap>()?;
    m.add_class::<Block>()?;
    m.add_function(wrap_pyfunction!(publish, m)?)?;
    m.add_function(wrap_pyfunction!(peek_generation, m)?)?;
    m.add_function(wrap_pyfunction!(format_version, m)?)?;
    Ok(())
}
