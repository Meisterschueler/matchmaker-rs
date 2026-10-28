//! Typed errors of the library (constitution IV).
//!
//! Every error names what failed and with which input. No public function panics on user input.

use std::path::PathBuf;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors returned by `matchmaker_rs`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A file could not be read or written.
    #[error("cannot access {path}: {source}")]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The file extension is not supported for this kind of input.
    #[error("unsupported format {extension:?} for {path}")]
    UnsupportedFormat {
        /// File that was given.
        path: PathBuf,
        /// Its extension (possibly empty).
        extension: String,
    },
    /// A MusicXML (or `.mxl`) file could not be parsed.
    #[error("invalid MusicXML in {path}: {message}")]
    MusicXml {
        /// File that failed.
        path: PathBuf,
        /// What was wrong.
        message: String,
    },
    /// A MIDI file could not be parsed.
    #[error("invalid MIDI in {path}: {message}")]
    Midi {
        /// File that failed.
        path: PathBuf,
        /// What was wrong.
        message: String,
    },
    /// The score was parsed but cannot be used (e.g. no notes).
    #[error("invalid score {path}: {reason}")]
    InvalidScore {
        /// Score file.
        path: PathBuf,
        /// Why it cannot be used.
        reason: String,
    },
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
