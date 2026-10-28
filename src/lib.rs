//! Pure-Rust port of [pymatchmaker](https://github.com/pymatchmaker/matchmaker): real-time
//! music alignment (score following) for MIDI performances.
//!
//! The pipeline mirrors pymatchmaker: `Stream` → `Processor` → `OnlineAlignment`.
//! Every module names the Python code it ports; results are verified against the reference
//! release with golden tests (see `tests/golden/`).

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
