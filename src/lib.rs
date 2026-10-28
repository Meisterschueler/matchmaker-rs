//! Pure-Rust port of [pymatchmaker](https://github.com/pymatchmaker/matchmaker): real-time
//! music alignment (score following) for MIDI performances.
//!
//! The pipeline mirrors pymatchmaker: `Stream` → `Processor` → `OnlineAlignment`.
//! Every module names the Python code it ports; results are verified against the reference
//! release with golden tests (see `tests/golden/`).

pub mod error;
pub mod features;
pub mod io;
pub mod performance;
pub mod score;

pub use error::{Error, Result};
