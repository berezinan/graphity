//! Snapshot format core: encoder and memory-mapped reader.

pub mod blocks;
pub mod encode;
pub mod graph;
pub mod header;
pub mod pool;
pub mod reader;
pub mod writer;

pub use encode::Encoder;
pub use graph::Graph;
pub use header::{DictRef, Header, HeaderMeta, SectionKind, SnapError, SourceInfo, FORMAT_VERSION};
pub use reader::Snapshot;
pub use writer::{publish, SnapWriter};
