//! Task 6.2: minimal native reader. Start, open the snapshot, answer one
//! `query`, exit. Measures process start + open + first answer without a
//! Python interpreter (D11).
//!
//!     snap-query <graph.snap> <fingerprint-hex> "<question>"   seeds + scores to stdout
//!     snap-query --terms "<question>"                          tokenisation only
//!
//! Tokenisation mirrors `serve._query_terms` and `serve._search_tokens`; it is
//! checked against them on the query set (README). Seeding is the combined
//! ranking only (`_score_query` without `_pick_seeds`), which is enough to
//! time the read path; the output is not the rendered `query` answer.

use snap_core::graph::Graph;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

const STOPWORDS: &str = include_str!("stopwords.txt");

fn is_word(c: char) -> bool {
    // Python `\w` for str patterns: alphanumeric or underscore.
    c.is_alphanumeric() || c == '_'
}

fn is_chinese(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn searchable(t: &str) -> bool {
    if t.chars().all(|c| c.is_ascii_lowercase()) {
        return t.chars().count() > 2;
    }
    // spike_patches.is_searchable: one- and two-letter Cyrillic words drop out.
    let n = t.chars().count();
    if n > 0 && n <= 2 && t.chars().all(|c| ('\u{0400}'..='\u{052f}').contains(&c)) {
        return false;
    }
    true
}

/// `serve._query_terms` (no jieba installed: Chinese falls back to bigrams).
pub fn query_terms(q: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for raw in q.split_whitespace() {
        let low = raw.to_lowercase();
        if raw.chars().any(is_chinese) {
            let s: Vec<char> = low.trim().chars().collect();
            let mut segs: Vec<String> = if s.len() > 1 {
                (0..s.len() - 1).map(|i| s[i..i + 2].iter().collect()).collect()
            } else {
                vec![s.iter().collect()]
            };
            let whole: String = s.iter().collect();
            if s.len() > 1 && !segs.contains(&whole) {
                segs.push(whole);
            }
            for seg in segs {
                let seg = seg.trim().to_string();
                if !seg.is_empty() && searchable(&seg) {
                    terms.push(seg);
                }
            }
        } else {
            let mut cur = String::new();
            for c in low.chars().chain(std::iter::once(' ')) {
                if is_word(c) {
                    cur.push(c);
                } else if !cur.is_empty() {
                    if searchable(&cur) {
                        terms.push(cur.clone());
                    }
                    cur.clear();
                }
            }
        }
    }
    let stop: HashSet<&str> = STOPWORDS.lines().collect();
    let content: Vec<String> = terms.iter().filter(|t| !stop.contains(t.as_str())).cloned().collect();
    if content.is_empty() { terms } else { content }
}

/// `serve._search_tokens`: NFKD, drop combining marks, lower, split on `[^\W_]+`.
pub fn search_tokens(t: &str) -> Vec<String> {
    let folded: String = t.nfkd().filter(|c| !is_combining_mark(*c)).collect::<String>().to_lowercase();
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in folded.chars().chain(std::iter::once(' ')) {
        if is_word(c) && c != '_' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    out
}

fn hex32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("fingerprint hex");
    }
    out
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--terms") {
        let terms = query_terms(&args[1]);
        let mut norm: Vec<String> = Vec::new();
        for t in &terms {
            for tok in search_tokens(t) {
                if !norm.contains(&tok) {
                    norm.push(tok);
                }
            }
        }
        println!("{}\t{}", terms.join(" "), norm.join(" "));
        return;
    }
    let (path, fp, question) = (&args[0], hex32(&args[1]), &args[2]);
    let g = Graph::open(Path::new(path), &fp).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let t_open = t0.elapsed();
    let mut norm: Vec<String> = Vec::new();
    for t in query_terms(question) {
        for tok in search_tokens(&t) {
            if !norm.contains(&tok) {
                norm.push(tok);
            }
        }
    }
    let terms: Vec<&[u8]> = norm.iter().map(|s| s.as_bytes()).collect();
    let n = g.n.max(1) as f64;
    let df = g.df(&terms);
    let w: Vec<f64> = df.iter().map(|&d| (1.0 + n / (1.0 + d as f64)).ln()).collect();
    let joined = norm.join(" ");
    // `max((idf[t] for t in norm_terms), default=1.0)`
    let jw = if w.is_empty() { 1.0 } else { w.iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
    let k = norm.len().max(1) as f64;
    let cov: Vec<f64> = (0..=norm.len()).map(|m| (m as f64 / k).powi(2)).collect();
    let out = g.score(&terms, &w, joined.as_bytes(), jw, &cov, false);
    let t_all = t0.elapsed();
    for (i, s) in out.idx.iter().zip(&out.score).take(5) {
        println!("{s}\t{}", String::from_utf8_lossy(&g.id(*i as usize)));
    }
    eprintln!(
        "open_ms={:.2} total_ms={:.2} nodes={} hits={}",
        t_open.as_secs_f64() * 1000.0,
        t_all.as_secs_f64() * 1000.0,
        g.n,
        out.idx.len()
    );
}
