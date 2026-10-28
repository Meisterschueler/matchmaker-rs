//! Pure-Rust port of [pymatchmaker](https://github.com/pymatchmaker/matchmaker): real-time
//! music alignment (score following) for MIDI performances.
//!
//! The pipeline mirrors pymatchmaker: `Stream` → `Processor` → `OnlineAlignment`.
//! Every module names the Python code it ports; results are verified against the reference
//! release with golden tests (see `tests/golden/`).
//!
//! ```
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?;
//! let positions = mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;
//! assert!(!positions.is_empty());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

pub mod base;
pub mod error;
pub mod export;
pub mod features;
pub mod io;
pub mod matchmaker;
pub mod performance;
pub mod prob;
pub mod score;
pub mod utils;

pub use error::{Error, Result};
