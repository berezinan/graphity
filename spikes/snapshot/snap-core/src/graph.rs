//! Read-side primitives over a mapped snapshot (D6, D13).
//!
//! Opening parses the header and the block directories only; nothing here
//! walks all nodes at open time.

use crate::blocks::{cast, parse};
use crate::encode::{S_NL, SEARCH};
use crate::header::{SectionKind, SnapError};
use crate::pool::{FcPool, Mode, Pool, Postings};
use crate::reader::Snapshot;
use memchr::memmem;
use std::collections::HashMap;
use std::path::Path;

pub struct Graph {
    pub snap: Snapshot,
    blocks: HashMap<String, (usize, usize)>,
    pub n: usize,
    pub e: usize,
}

pub struct ScoreOut {
    pub idx: Vec<u32>,
    pub score: Vec<f64>,
    /// Per term: best singleton (score, node) by `_score_query`'s key.
    pub best: Vec<Option<(f64, u32)>>,
}

impl Graph {
    pub fn open(path: &Path, fingerprint: &[u8; 32]) -> Result<Graph, SnapError> {
        let snap = Snapshot::open(path, fingerprint)?;
        let mut blocks = HashMap::new();
        for s in &snap.header.sections {
            parse(snap.bytes(), s.offset as usize, s.len as usize, &mut blocks)?;
        }
        for kind in [SectionKind::Dict, SectionKind::Nodes, SectionKind::Edges, SectionKind::Adj] {
            if snap.header.section(kind).is_none() {
                return Err(SnapError::Corrupt(format!("section {kind:?} missing")));
            }
        }
        let n = snap.header.meta.node_count as usize;
        let e = snap.header.meta.edge_count as usize;
        let g = Graph { snap, blocks, n, e };
        if g.u32s("n.degt").len() != n || g.u32s("l.src").len() != e {
            return Err(SnapError::Corrupt("counts differ from header".into()));
        }
        Ok(g)
    }

    // ---- raw access ---------------------------------------------------------
    pub fn block(&self, name: &str) -> Option<&[u8]> {
        let &(o, l) = self.blocks.get(name)?;
        Some(&self.snap.bytes()[o..o + l])
    }

    fn need(&self, name: &str) -> &[u8] {
        self.block(name).unwrap_or_else(|| panic!("snapshot block {name} missing"))
    }

    pub fn u32s(&self, name: &str) -> &[u32] {
        cast(self.need(name))
    }

    pub fn u64s(&self, name: &str) -> &[u64] {
        cast(self.need(name))
    }

    pub fn pool(&self, name: &str) -> Option<Pool<'_>> {
        Some(Pool { blob: self.block(&format!("{name}.blob"))?, off: cast(self.block(&format!("{name}.off"))?) })
    }

    fn postings(&self, name: &str) -> Postings<'_> {
        Postings { off: self.u64s(&format!("{name}.poff")), nodes: self.u32s(&format!("{name}.post")) }
    }

    fn fc(&self, name: &str) -> FcPool<'_> {
        FcPool { blob: self.need(&format!("{name}.fc")), off: cast(self.need(&format!("{name}.fco"))) }
    }

    pub fn id(&self, i: usize) -> Vec<u8> {
        self.fc("ids").get(i)
    }

    fn key_lower(&self, i: usize) -> Vec<u8> {
        match self.u32s("n.idl")[i] {
            0 => self.id(i),
            c => self.fc("idl").get(c as usize - 1),
        }
    }

    fn key_norm(&self, i: usize) -> Vec<u8> {
        match self.u32s("n.idn")[i] {
            0 => self.key_lower(i),
            c => self.fc("idn").get(c as usize - 1),
        }
    }

    /// Search string `f` of node `i` ("" when empty).
    pub fn sget(&self, f: usize, i: usize) -> &[u8] {
        match self.u32s(&format!("n.s.{}", SEARCH[f]))[i] {
            0 => b"",
            c => self.pool(&format!("s.{}", SEARCH[f])).unwrap().get(c as usize - 1),
        }
    }

    // ---- node lookup ----------------------------------------------------------
    /// Nodes whose search string `field` matches any needle, ascending.
    pub fn match_nodes(&self, field: &str, mode: Mode, needles: &[&[u8]]) -> Vec<u32> {
        let f = SEARCH.iter().position(|s| *s == field).expect("unknown search field");
        let mut bits = vec![0u64; self.n.div_ceil(64)];
        self.mark(f, mode, needles, &mut bits);
        drain(&bits)
    }

    fn mark(&self, f: usize, mode: Mode, needles: &[&[u8]], bits: &mut [u64]) {
        let name = format!("s.{}", SEARCH[f]);
        let pool = self.pool(&name).unwrap();
        let post = self.postings(&name);
        for nd in needles {
            for j in pool.matching(nd, mode) {
                for &i in post.of(j as usize) {
                    bits[i as usize / 64] |= 1 << (i % 64);
                }
            }
        }
    }

    /// Nodes whose lower-cased (`kind = "lower"`) or diacritic-folded
    /// (`"norm"`) id equals / starts with the needle, ascending.
    pub fn match_ids(&self, kind: &str, mode: Mode, needles: &[&[u8]]) -> Vec<u32> {
        let mut bits = vec![0u64; self.n.div_ceil(64)];
        self.mark_ids(kind, mode, needles, &mut bits);
        drain(&bits)
    }

    fn mark_ids(&self, kind: &str, mode: Mode, needles: &[&[u8]], bits: &mut [u64]) {
        let lower = match kind {
            "lower" => true,
            "norm" => false,
            _ => panic!("unknown id kind {kind}"),
        };
        let perm = self.u32s(if lower { "perm.lower" } else { "perm.norm" });
        let key = |i: usize| if lower { self.key_lower(i) } else { self.key_norm(i) };
        for nd in needles {
            let lo = perm.partition_point(|&i| key(i as usize).as_slice() < *nd);
            for &i in &perm[lo..] {
                let k = key(i as usize);
                let hit = match mode {
                    Mode::Equal => k.as_slice() == *nd,
                    Mode::Prefix => k.starts_with(nd),
                    Mode::Substring | Mode::Bare => panic!("ids support equal / prefix only"),
                };
                if !hit {
                    break;
                }
                bits[i as usize / 64] |= 1 << (i % 64);
            }
        }
    }

    // ---- scoring (mirror of serve._score_query) --------------------------------
    /// Document frequency for `_compute_idf`: nodes whose lower-cased
    /// `norm_label` contains the term.
    pub fn df(&self, terms: &[&[u8]]) -> Vec<u64> {
        let nl = self.pool("s.nl").unwrap();
        let nl_post = self.postings("s.nl");
        let nli = self.pool("s.nli").unwrap();
        let nli_post = self.postings("s.nli");
        let has_nli = nli.len() > 0;
        let nli_col = self.u32s("n.s.nli");
        terms
            .iter()
            .map(|t| {
                let mut c = 0u64;
                for j in nl.matching(t, Mode::Substring) {
                    let p = nl_post.of(j as usize);
                    c += if has_nli { p.iter().filter(|&&i| nli_col[i as usize] == 0).count() as u64 } else { p.len() as u64 };
                }
                if has_nli {
                    for j in nli.matching(t, Mode::Substring) {
                        c += nli_post.of(j as usize).len() as u64;
                    }
                }
                c
            })
            .collect()
    }

    /// `_score_query` over the snapshot. `w[k]` is `idf.get(terms[k], 1.0)`,
    /// `cov[m]` is `(m / n_terms) ** 2` -- both computed in Python so the
    /// float values are CPython's.
    /// Search column `f` resolved once: per-node codes and the string pool.
    fn scol(&self, f: usize) -> (&[u32], Pool<'_>) {
        (self.u32s(&format!("n.s.{}", SEARCH[f])), self.pool(&format!("s.{}", SEARCH[f])).unwrap())
    }

    pub fn score(&self, terms: &[&[u8]], w: &[f64], joined: &[u8], jw: f64, cov: &[f64], collect: bool) -> ScoreOut {
        let words = self.n.div_ceil(64);
        let mut bits = vec![0u64; words];
        let mut needles: Vec<&[u8]> = terms.to_vec();
        if !joined.is_empty() {
            needles.push(joined);
        }
        for f in [S_NL, 2, 3, 5, 6, 7] {
            self.mark(f, Mode::Substring, &needles, &mut bits);
        }
        // `nid.lower() == x` for each needle, from the sorted id permutation:
        // no id is decoded in the loop below.
        let id_eq: Vec<Vec<u64>> = needles
            .iter()
            .map(|nd| {
                let mut b = vec![0u64; words];
                self.mark_ids("lower", Mode::Equal, &[*nd], &mut b);
                for (x, y) in bits.iter_mut().zip(&b) {
                    *x |= *y;
                }
                b
            })
            .collect();
        let has = |b: &[u64], i: usize| b[i / 64] >> (i % 64) & 1 == 1;
        let joined_k = terms.len();

        let finders: Vec<memmem::Finder> = terms.iter().map(|t| memmem::Finder::new(t)).collect();
        let col = |f: usize| self.scol(f);
        let (nl_c, nl_p) = col(S_NL);
        let (lt_c, lt_p) = col(2);
        let (src_c, src_p) = col(3);
        let (rat_c, rat_p) = col(5);
        let (atn_c, atn_p) = col(6);
        let (att_c, att_p) = col(7);
        fn get<'a>(c: &[u32], p: &Pool<'a>, i: usize) -> &'a [u8] {
            match c[i] {
                0 => b"",
                x => p.get(x as usize - 1),
            }
        }
        let degq = self.u32s("n.degq");
        let llen = self.u32s("n.llen");
        let rank = self.u32s("n.idrank");
        let contains = |hay: &[u8], k: usize| finders[k].find(hay).is_some();
        let mut scored: Vec<(f64, u32)> = vec![];
        let mut best: Vec<Option<(f64, u32)>> = vec![None; terms.len()];
        for i in drain(&bits) {
            let i_us = i as usize;
            let nl = get(nl_c, &nl_p, i_us);
            let bare = rstrip_parens(nl);
            let lt = get(lt_c, &lt_p, i_us);
            let src = get(src_c, &src_p, i_us);
            let rat = get(rat_c, &rat_p, i_us);
            let atn = get(atn_c, &atn_p, i_us);
            let att = get(att_c, &att_p, i_us);
            let mut score = 0.0f64;
            if !joined.is_empty() {
                if joined == nl || joined == bare || joined == lt || has(&id_eq[joined_k], i_us) {
                    score += 1000.0 * 10.0 * jw;
                } else if nl.starts_with(joined) || bare.starts_with(joined) || lt.starts_with(joined) {
                    score += 100.0 * 10.0 * jw;
                }
            }
            let mut matched = 0usize;
            let mut tiered = 0.0f64;
            for (k, t) in terms.iter().enumerate() {
                let t = *t;
                let wk = w[k];
                let mut tier_value = 0.0f64;
                let mut substr_value = 0.0f64;
                let mut source_value = 0.0f64;
                if t == nl || t == bare {
                    tier_value = 1000.0 * wk;
                    matched += 1;
                } else if nl.starts_with(t) || bare.starts_with(t) {
                    tier_value = 100.0 * wk;
                    matched += 1;
                } else if contains(nl, k) {
                    substr_value = 1.0 * wk;
                    score += substr_value;
                    matched += 1;
                }
                if contains(src, k) {
                    source_value = 0.5 * wk;
                    score += source_value;
                }
                let mut rationale_value = 0.0f64;
                if !rat.is_empty() && contains(rat, k) {
                    rationale_value = 0.75 * wk;
                    score += rationale_value;
                }
                let mut attr_value = 0.0f64;
                if !atn.is_empty() && (contains(atn, k) || contains(att, k)) {
                    attr_value = 0.75 * wk;
                    score += attr_value;
                }
                tiered += tier_value;
                if collect {
                    let mut singleton = if t == nl || t == bare || t == lt || has(&id_eq[k], i_us) {
                        1000.0 * 10.0 * wk
                    } else if nl.starts_with(t) || bare.starts_with(t) || lt.starts_with(t) {
                        100.0 * 10.0 * wk
                    } else {
                        0.0
                    };
                    singleton = singleton + tier_value + substr_value + source_value + rationale_value + attr_value;
                    if singleton > 0.0 {
                        // key (-singleton, -degree, label_len, nid): smallest wins
                        let better = match best[k] {
                            None => true,
                            Some((s, j)) => {
                                let ju = j as usize;
                                key_less(
                                    (singleton, degq[i_us], llen[i_us], rank[i_us]),
                                    (s, degq[ju], llen[ju], rank[ju]),
                                )
                            }
                        };
                        if better {
                            best[k] = Some((singleton, i));
                        }
                    }
                }
            }
            if tiered != 0.0 {
                score += tiered * cov[matched];
            }
            if score > 0.0 {
                scored.push((score, i));
            }
        }
        // (-score, len(label or nid), nid)
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(llen[a.1 as usize].cmp(&llen[b.1 as usize]))
                .then(rank[a.1 as usize].cmp(&rank[b.1 as usize]))
        });
        ScoreOut { idx: scored.iter().map(|x| x.1).collect(), score: scored.iter().map(|x| x.0).collect(), best }
    }

    // ---- traversal graph (undirected, networkx order) ---------------------------
    fn passes(&self, p: u32, filt: Option<&[u32]>) -> bool {
        match filt {
            None => true,
            Some(f) => {
                let c = self.u32s("p.ctx")[p as usize];
                c != 0 && f.contains(&c)
            }
        }
    }

    pub fn tneighbors(&self, i: usize, filt: Option<&[u32]>) -> Vec<u32> {
        let off = self.u64s("t.off");
        let (a, b) = (off[i] as usize, off[i + 1] as usize);
        let nbr = &self.u32s("t.nbr")[a..b];
        match filt {
            None => nbr.to_vec(),
            Some(_) => {
                let pair = &self.u32s("t.pair")[a..b];
                let rank = self.u32s("p.rank");
                let mut v: Vec<(u32, u32)> =
                    (0..nbr.len()).filter(|&k| self.passes(pair[k], filt)).map(|k| (rank[pair[k] as usize], nbr[k])).collect();
                v.sort_unstable();
                v.into_iter().map(|x| x.1).collect()
            }
        }
    }

    pub fn tdegree(&self, i: usize, filt: Option<&[u32]>) -> u32 {
        if filt.is_none() {
            return self.u32s("n.degt")[i];
        }
        let off = self.u64s("t.off");
        let (a, b) = (off[i] as usize, off[i + 1] as usize);
        let nbr = self.u32s("t.nbr");
        let pair = self.u32s("t.pair");
        let mut d = 0u32;
        for k in a..b {
            if self.passes(pair[k], filt) {
                d += if nbr[k] as usize == i { 2 } else { 1 };
            }
        }
        d
    }

    pub fn hub_threshold(&self, filt: Option<&[u32]>) -> u32 {
        match filt {
            None => self.u32s("hub")[0],
            Some(_) => crate::encode::hub_threshold((0..self.n).map(|i| self.tdegree(i, filt)), self.n),
        }
    }

    /// Links of the traversal pair {i, j}, in merge order; empty if not adjacent.
    pub fn pair_links(&self, i: usize, j: usize) -> Vec<u32> {
        let off = self.u64s("t.off");
        let nbr = self.u32s("t.nbr");
        let pair = self.u32s("t.pair");
        for k in off[i] as usize..off[i + 1] as usize {
            if nbr[k] as usize == j {
                let p = pair[k] as usize;
                let loff = self.u64s("p.loff");
                return self.u32s("p.links")[loff[p] as usize..loff[p + 1] as usize].to_vec();
            }
        }
        vec![]
    }

    // ---- directed view ----------------------------------------------------------
    pub fn out_links(&self, i: usize) -> &[u32] {
        let off = self.u64s("o.off");
        &self.u32s("o.link")[off[i] as usize..off[i + 1] as usize]
    }

    pub fn in_links(&self, i: usize) -> &[u32] {
        let off = self.u64s("i.off");
        &self.u32s("i.link")[off[i] as usize..off[i + 1] as usize]
    }

    pub fn deg_dir(&self, i: usize) -> u32 {
        (self.out_links(i).len() + self.in_links(i).len()) as u32
    }

    /// Every link joining i and j, either direction.
    pub fn links_between(&self, i: usize, j: usize) -> Vec<u32> {
        let tgt = self.u32s("l.tgt");
        let mut v: Vec<u32> = self.out_links(i).iter().copied().filter(|&l| tgt[l as usize] as usize == j).collect();
        if i != j {
            v.extend(self.out_links(j).iter().copied().filter(|&l| tgt[l as usize] as usize == i));
        }
        v
    }

    /// One BFS step of `_lexmin_shortest_path` over `serve._path_search_graph`:
    /// stored direction plus every `contains` walked back (#3878).
    pub fn path_expand(&self, frontier: &[u32], forward: bool, undirected: bool, contains: u32) -> Vec<Vec<u32>> {
        let src = self.u32s("l.src");
        let tgt = self.u32s("l.tgt");
        let rel = self.u32s("l.v.relation");
        frontier
            .iter()
            .map(|&u| {
                let u = u as usize;
                let mut out = vec![];
                for &l in self.out_links(u) {
                    let l = l as usize;
                    if undirected || forward || (contains != 0 && rel[l] == contains) {
                        out.push(tgt[l]);
                    }
                }
                for &l in self.in_links(u) {
                    let l = l as usize;
                    if undirected || !forward || (contains != 0 && rel[l] == contains) {
                        out.push(src[l]);
                    }
                }
                out
            })
            .collect()
    }

    // ---- records ----------------------------------------------------------------
    pub fn codes(&self, prefix: &str, fields: &[String], i: usize) -> Vec<u32> {
        fields.iter().map(|f| self.u32s(&format!("{prefix}{f}"))[i]).collect()
    }

    pub fn cold(&self, kind: &str, i: usize) -> &[u8] {
        let name = if kind == "n" { "cn" } else { "cl" };
        Pool { blob: self.need(&format!("{name}.blob")), off: cast(self.need(&format!("{name}.off"))) }.get(i)
    }
}

fn key_less(a: (f64, u32, u32, u32), b: (f64, u32, u32, u32)) -> bool {
    // (-singleton, -degree, label_len, nid): a < b
    if a.0 != b.0 {
        return a.0 > b.0;
    }
    if a.1 != b.1 {
        return a.1 > b.1;
    }
    if a.2 != b.2 {
        return a.2 < b.2;
    }
    a.3 < b.3
}

fn rstrip_parens(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && (s[end - 1] == b'(' || s[end - 1] == b')') {
        end -= 1;
    }
    &s[..end]
}

fn drain(bits: &[u64]) -> Vec<u32> {
    let mut out = vec![];
    for (w, &b) in bits.iter().enumerate() {
        let mut b = b;
        while b != 0 {
            let t = b.trailing_zeros();
            out.push((w * 64) as u32 + t);
            b &= b - 1;
        }
    }
    out
}
