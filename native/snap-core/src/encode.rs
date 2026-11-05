//! Snapshot encoder (D3, D5, D13). Python hands over already-normalised
//! columns; this module interns strings, derives the networkx-ordered
//! topology and the search postings, and lays the sections out.
//!
//! Value columns are opaque to Rust: each cell is the JSON text of one
//! attribute value (0 = attribute absent), decoded by Python. Rust only reads
//! the search strings, the endpoints and the code of the `context` value.

use crate::blocks::{BlocksBuilder, Data};
use crate::header::{HeaderMeta, SectionKind, SnapError};
use crate::pool::{csr, front_code, PoolBuilder};
use crate::writer::SnapWriter;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::time::Instant;

/// Search columns, in this order. `nli` is the IDF form of `nl`
/// (`_compute_idf` lower-cases the stored `norm_label`); 0 means "same as nl".
pub const SEARCH: [&str; 8] = ["nl", "nli", "lt", "src", "st", "rat", "atn", "att"];
pub const S_NL: usize = 0;
pub const S_NLI: usize = 1;

pub struct Encoder {
    pub directed: bool,
    node_fields: Vec<String>,
    link_fields: Vec<String>,
    ids: PoolBuilder,
    idl: PoolBuilder,
    n_idl: Vec<u32>,
    idn: PoolBuilder,
    n_idn: Vec<u32>,
    search: Vec<PoolBuilder>,
    n_search: Vec<Vec<u32>>,
    nvals: Vec<PoolBuilder>,
    n_vals: Vec<Vec<u32>>,
    lvals: Vec<PoolBuilder>,
    l_vals: Vec<Vec<u32>>,
    llen: Vec<u32>,
    ncold: PoolBuilder,
    lcold: PoolBuilder,
    src: Vec<u32>,
    tgt: Vec<u32>,
}

fn code(p: &mut PoolBuilder, v: Option<&str>) -> Result<u32, SnapError> {
    Ok(match v {
        None => 0,
        Some(s) => p.push(s.as_bytes())? + 1,
    })
}

impl Encoder {
    pub fn new(node_fields: Vec<String>, link_fields: Vec<String>, directed: bool) -> Self {
        let nf = node_fields.len();
        let lf = link_fields.len();
        Encoder {
            directed,
            ids: PoolBuilder::new(false),
            idl: PoolBuilder::new(false),
            n_idl: vec![],
            idn: PoolBuilder::new(false),
            n_idn: vec![],
            search: (0..SEARCH.len()).map(|_| PoolBuilder::new(true)).collect(),
            n_search: vec![vec![]; SEARCH.len()],
            nvals: (0..nf).map(|_| PoolBuilder::new(true)).collect(),
            n_vals: vec![vec![]; nf],
            lvals: (0..lf).map(|_| PoolBuilder::new(true)).collect(),
            l_vals: vec![vec![]; lf],
            llen: vec![],
            ncold: PoolBuilder::new(false),
            lcold: PoolBuilder::new(false),
            src: vec![],
            tgt: vec![],
            node_fields,
            link_fields,
        }
    }

    pub fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// One batch of nodes. `search[f][i]` is "" when empty; `vals[f][i]` is the
    /// JSON text of field `f`, `None` when the node lacks it.
    #[allow(clippy::too_many_arguments)]
    pub fn add_nodes(
        &mut self,
        ids: &[String],
        idl: &[Option<String>],
        idn: &[Option<String>],
        search: &[Vec<String>],
        vals: &[Vec<Option<String>>],
        llen: &[u32],
        cold: &[Option<String>],
    ) -> Result<(), SnapError> {
        let n = ids.len();
        let ok = idl.len() == n
            && idn.len() == n
            && llen.len() == n
            && cold.len() == n
            && search.len() == SEARCH.len()
            && search.iter().all(|c| c.len() == n)
            && vals.len() == self.node_fields.len()
            && vals.iter().all(|c| c.len() == n);
        if !ok {
            return Err(SnapError::Corrupt("add_nodes: column lengths differ".into()));
        }
        for i in 0..n {
            self.ids.push(ids[i].as_bytes())?;
            self.n_idl.push(code(&mut self.idl, idl[i].as_deref())?);
            self.n_idn.push(code(&mut self.idn, idn[i].as_deref())?);
            for f in 0..SEARCH.len() {
                let s = &search[f][i];
                let c = if s.is_empty() { 0 } else { self.search[f].push(s.as_bytes())? + 1 };
                self.n_search[f].push(c);
            }
            for f in 0..self.node_fields.len() {
                let c = code(&mut self.nvals[f], vals[f][i].as_deref())?;
                self.n_vals[f].push(c);
            }
            self.llen.push(llen[i]);
            self.ncold.push(cold[i].as_deref().unwrap_or("").as_bytes())?;
        }
        Ok(())
    }

    pub fn add_links(
        &mut self,
        src: &[u32],
        tgt: &[u32],
        vals: &[Vec<Option<String>>],
        cold: &[Option<String>],
    ) -> Result<(), SnapError> {
        let n = src.len();
        let ok = tgt.len() == n
            && cold.len() == n
            && vals.len() == self.link_fields.len()
            && vals.iter().all(|c| c.len() == n);
        if !ok {
            return Err(SnapError::Corrupt("add_links: column lengths differ".into()));
        }
        let nn = self.node_count() as u32;
        for i in 0..n {
            if src[i] >= nn || tgt[i] >= nn {
                return Err(SnapError::Corrupt("link endpoint out of range".into()));
            }
            self.src.push(src[i]);
            self.tgt.push(tgt[i]);
            for f in 0..self.link_fields.len() {
                let c = code(&mut self.lvals[f], vals[f][i].as_deref())?;
                self.l_vals[f].push(c);
            }
            self.lcold.push(cold[i].as_deref().unwrap_or("").as_bytes())?;
        }
        Ok(())
    }

    fn key_lower(&self, i: usize) -> &[u8] {
        match self.n_idl[i] {
            0 => self.ids.get(i),
            c => self.idl.get(c as usize - 1),
        }
    }

    fn key_norm(&self, i: usize) -> &[u8] {
        match self.n_idn[i] {
            0 => self.key_lower(i),
            c => self.idn.get(c as usize - 1),
        }
    }

    /// Build topology and indexes, write the file. Returns phase timings in seconds.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        mut self,
        path: &Path,
        meta: HeaderMeta,
        meta_json: &str,
        stats_json: &str,
        graph_json: &str,
        comm_off: Vec<u64>,
        comm_mem: Vec<u32>,
    ) -> Result<Vec<(&'static str, f64)>, SnapError> {
        let mut times = vec![];
        let n = self.node_count();
        let e = self.src.len();
        let mut meta = meta;
        meta.node_count = n as u64;
        meta.edge_count = e as u64;
        let t0 = Instant::now();

        // ---- topology (D4, D13) -------------------------------------------
        // L: the order networkx inserts pairs. An undirected file is loaded
        // straight from `links`; a directed one goes through DiGraph, whose
        // `edges()` walk sources in node order (stable within a source), and
        // `_traversal_view` re-inserts in that order.
        let lorder: Vec<u32> = if self.directed {
            let (_, v) = csr(&self.src, n);
            v
        } else {
            (0..e as u32).collect()
        };
        let mut pair_of: FxHashMap<u64, u32> = FxHashMap::default();
        let (mut pa, mut pb) = (Vec::<u32>::new(), Vec::<u32>::new());
        let mut link_pair = vec![0u32; e];
        for &l in &lorder {
            let (u, v) = (self.src[l as usize], self.tgt[l as usize]);
            let key = if u <= v { (u as u64) << 32 | v as u64 } else { (v as u64) << 32 | u as u64 };
            let p = *pair_of.entry(key).or_insert_with(|| {
                pa.push(u);
                pb.push(v);
                (pa.len() - 1) as u32
            });
            link_pair[l as usize] = p;
        }
        drop(pair_of);
        let np = pa.len();
        // Links of each pair, in L order (attributes merge by dict.update).
        let lp_in_order: Vec<u32> = lorder.iter().map(|&l| link_pair[l as usize]).collect();
        let (p_loff, pos) = csr(&lp_in_order, np);
        let p_links: Vec<u32> = pos.iter().map(|&k| lorder[k as usize]).collect();
        drop(lp_in_order);
        // Adjacency of the undirected traversal graph: per node, pairs in
        // order of first appearance.
        let mut deg = vec![0u64; n + 1];
        for p in 0..np {
            deg[pa[p] as usize + 1] += 1;
            if pa[p] != pb[p] {
                deg[pb[p] as usize + 1] += 1;
            }
        }
        for i in 0..n {
            deg[i + 1] += deg[i];
        }
        let t_off = deg;
        let mut fill = t_off.clone();
        let mut t_nbr = vec![0u32; t_off[n] as usize];
        let mut t_pair = vec![0u32; t_off[n] as usize];
        for p in 0..np {
            let (u, v) = (pa[p] as usize, pb[p] as usize);
            t_nbr[fill[u] as usize] = v as u32;
            t_pair[fill[u] as usize] = p as u32;
            fill[u] += 1;
            if u != v {
                t_nbr[fill[v] as usize] = u as u32;
                t_pair[fill[v] as usize] = p as u32;
                fill[v] += 1;
            }
        }
        drop(fill);
        // Rank of each pair in `G.edges()` order: nodes in order, each pair
        // reported at its first endpoint. `_filter_graph_by_context` re-adds
        // edges in this order, so a filtered neighbour list is rank order.
        let mut p_rank = vec![0u32; np];
        let mut r = 0u32;
        let mut deg_t = vec![0u32; n];
        for x in 0..n {
            let (a, b) = (t_off[x] as usize, t_off[x + 1] as usize);
            let mut self_loop = 0;
            for k in a..b {
                let y = t_nbr[k] as usize;
                if y == x {
                    self_loop = 1;
                }
                if y >= x {
                    p_rank[t_pair[k] as usize] = r;
                    r += 1;
                }
            }
            deg_t[x] = (b - a) as u32 + self_loop;
        }
        // Merged `context` of each pair: the last link in L order that has one.
        let ctx_col = self.link_fields.iter().position(|f| f == "context");
        let mut p_ctx = vec![0u32; np];
        if let Some(c) = ctx_col {
            for p in 0..np {
                for &l in &p_links[p_loff[p] as usize..p_loff[p + 1] as usize] {
                    let v = self.l_vals[c][l as usize];
                    if v != 0 {
                        p_ctx[p] = v;
                    }
                }
            }
        }
        drop(pa);
        drop(pb);
        // Directed view (explain, get_neighbors, path): arcs per endpoint in link order.
        let (o_off, o_link) = csr(&self.src, n);
        let (i_off, i_link) = csr(&self.tgt, n);
        let deg_q: Vec<u32> = if self.directed {
            (0..n).map(|x| ((o_off[x + 1] - o_off[x]) + (i_off[x + 1] - i_off[x])) as u32).collect()
        } else {
            deg_t.clone()
        };
        let hub = hub_threshold(deg_t.iter().copied(), n);
        times.push(("topology", t0.elapsed().as_secs_f64()));

        // ---- index -----------------------------------------------------------
        let t1 = Instant::now();
        let mut perm_lower: Vec<u32> = (0..n as u32).collect();
        perm_lower.sort_by(|&a, &b| self.key_lower(a as usize).cmp(self.key_lower(b as usize)));
        let mut perm_norm: Vec<u32> = (0..n as u32).collect();
        perm_norm.sort_by(|&a, &b| self.key_norm(a as usize).cmp(self.key_norm(b as usize)));
        // Rank of the raw id in byte (= code point) order: tie-breaks compare
        // ranks instead of decoding front-coded ids.
        let mut by_id: Vec<u32> = (0..n as u32).collect();
        by_id.sort_by(|&a, &b| self.ids.get(a as usize).cmp(self.ids.get(b as usize)));
        let mut id_rank = vec![0u32; n];
        for (r, &i) in by_id.iter().enumerate() {
            id_rank[i as usize] = r as u32;
        }
        drop(by_id);
        let mut postings = Vec::with_capacity(SEARCH.len());
        for f in 0..SEARCH.len() {
            let keys: Vec<u32> = self.n_search[f].iter().map(|&c| c.wrapping_sub(1)).collect();
            postings.push(csr(&keys, self.search[f].len()));
        }
        times.push(("index", t1.elapsed().as_secs_f64()));

        // ---- write -------------------------------------------------------------
        let t2 = Instant::now();
        for p in self.search.iter_mut().chain(self.nvals.iter_mut()).chain(self.lvals.iter_mut()) {
            p.seal();
        }
        let mut w = SnapWriter::create(path)?;

        let mut dict = BlocksBuilder::default();
        let pool = |b: &mut BlocksBuilder, name: &str, p: PoolBuilder| {
            b.put(format!("{name}.blob"), Data::Bytes(p.blob));
            b.put(format!("{name}.off"), Data::U64(p.off));
        };
        for (name, p) in [("ids", &self.ids), ("idl", &self.idl), ("idn", &self.idn)] {
            let (blob, off) = front_code(p);
            dict.put(format!("{name}.fc"), Data::Bytes(blob));
            dict.put(format!("{name}.fco"), Data::U64(off));
        }
        for (f, (p, (poff, post))) in std::mem::take(&mut self.search).into_iter().zip(postings).enumerate() {
            let name = format!("s.{}", SEARCH[f]);
            pool(&mut dict, &name, p);
            dict.put(format!("{name}.poff"), Data::U64(poff));
            dict.put(format!("{name}.post"), Data::U32(post));
        }
        for (f, p) in std::mem::take(&mut self.nvals).into_iter().enumerate() {
            pool(&mut dict, &format!("nv.{}", self.node_fields[f]), p);
        }
        for (f, p) in std::mem::take(&mut self.lvals).into_iter().enumerate() {
            pool(&mut dict, &format!("lv.{}", self.link_fields[f]), p);
        }
        w.begin_section(SectionKind::Dict)?;
        dict.write(&mut w)?;
        w.end_section()?;
        drop(dict);

        let mut nodes = BlocksBuilder::default();
        for (f, col) in std::mem::take(&mut self.n_vals).into_iter().enumerate() {
            nodes.put(format!("n.v.{}", self.node_fields[f]), Data::U32(col));
        }
        for (f, col) in std::mem::take(&mut self.n_search).into_iter().enumerate() {
            nodes.put(format!("n.s.{}", SEARCH[f]), Data::U32(col));
        }
        nodes.put("n.idl", Data::U32(std::mem::take(&mut self.n_idl)));
        nodes.put("n.idn", Data::U32(std::mem::take(&mut self.n_idn)));
        nodes.put("n.llen", Data::U32(std::mem::take(&mut self.llen)));
        nodes.put("n.idrank", Data::U32(id_rank));
        nodes.put("n.degq", Data::U32(deg_q));
        nodes.put("n.degt", Data::U32(deg_t));
        w.begin_section(SectionKind::Nodes)?;
        nodes.write(&mut w)?;
        w.end_section()?;
        drop(nodes);

        let mut edges = BlocksBuilder::default();
        edges.put("l.src", Data::U32(std::mem::take(&mut self.src)));
        edges.put("l.tgt", Data::U32(std::mem::take(&mut self.tgt)));
        for (f, col) in std::mem::take(&mut self.l_vals).into_iter().enumerate() {
            edges.put(format!("l.v.{}", self.link_fields[f]), Data::U32(col));
        }
        w.begin_section(SectionKind::Edges)?;
        edges.write(&mut w)?;
        w.end_section()?;
        drop(edges);

        let mut adj = BlocksBuilder::default();
        adj.put("t.off", Data::U64(t_off));
        adj.put("t.nbr", Data::U32(t_nbr));
        adj.put("t.pair", Data::U32(t_pair));
        adj.put("p.rank", Data::U32(p_rank));
        adj.put("p.ctx", Data::U32(p_ctx));
        adj.put("p.loff", Data::U64(p_loff));
        adj.put("p.links", Data::U32(p_links));
        adj.put("o.off", Data::U64(o_off));
        adj.put("o.link", Data::U32(o_link));
        adj.put("i.off", Data::U64(i_off));
        adj.put("i.link", Data::U32(i_link));
        w.begin_section(SectionKind::Adj)?;
        adj.write(&mut w)?;
        w.end_section()?;
        drop(adj);

        let mut index = BlocksBuilder::default();
        index.put("perm.lower", Data::U32(perm_lower));
        index.put("perm.norm", Data::U32(perm_norm));
        w.begin_section(SectionKind::Index)?;
        index.write(&mut w)?;
        w.end_section()?;

        let mut stats = BlocksBuilder::default();
        stats.put("meta.json", Data::Bytes(meta_json.as_bytes().to_vec()));
        stats.put("stats.json", Data::Bytes(stats_json.as_bytes().to_vec()));
        stats.put("hub", Data::U32(vec![hub]));
        stats.put("c.off", Data::U64(comm_off));
        stats.put("c.mem", Data::U32(comm_mem));
        w.begin_section(SectionKind::Stats)?;
        stats.write(&mut w)?;
        w.end_section()?;

        let mut cold = BlocksBuilder::default();
        let nc = std::mem::replace(&mut self.ncold, PoolBuilder::new(false));
        let lc = std::mem::replace(&mut self.lcold, PoolBuilder::new(false));
        cold.put("cn.blob", Data::Bytes(nc.blob));
        cold.put("cn.off", Data::U64(nc.off));
        cold.put("cl.blob", Data::Bytes(lc.blob));
        cold.put("cl.off", Data::U64(lc.off));
        cold.put("graph.json", Data::Bytes(graph_json.as_bytes().to_vec()));
        w.begin_section(SectionKind::Cold)?;
        cold.write(&mut w)?;
        w.end_section()?;
        times.push(("write", t2.elapsed().as_secs_f64()));

        let t3 = Instant::now();
        w.finish(meta)?;
        times.push(("flush", t3.elapsed().as_secs_f64()));
        Ok(times)
    }
}

/// `serve._hub_threshold`: p99 of the degree distribution, floored at 50.
pub fn hub_threshold(degrees: impl Iterator<Item = u32>, n: usize) -> u32 {
    if n == 0 {
        return 50;
    }
    let mut hist: Vec<u64> = vec![];
    for d in degrees {
        let d = d as usize;
        if d >= hist.len() {
            hist.resize(d + 1, 0);
        }
        hist[d] += 1;
    }
    // Python: int(len(degrees_sorted) * 0.99) -- float multiply, truncation.
    let idx = ((n as f64) * 0.99) as u64;
    let mut seen = 0u64;
    for (d, &c) in hist.iter().enumerate() {
        seen += c;
        if seen > idx {
            return (d as u32).max(50);
        }
    }
    50
}
