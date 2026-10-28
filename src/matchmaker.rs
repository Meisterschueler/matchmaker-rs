//! Port of `matchmaker.matchmaker.Matchmaker` for MIDI input: score + stream + processor +
//! follower, with the per-method defaults of the reference's `methods.yaml` (research R1).
//!
//! ```no_run
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?;
//! let positions = mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;
//! println!("{} positions, path of {}", positions.len(), mm.alignment_path().perf_time.len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::path::Path;
use std::str::FromStr;

use crate::base::{AlignmentPath, OnlineAlignment};
use crate::error::{Error, Result};
use crate::features::{Processor, pitch::PitchProcessor};
use crate::io::stream::{Frame, MidiMessage};
use crate::performance::Performance;
use crate::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
use crate::prob::outerhmm::{OuterProductHmm, OuterProductHmmConfig};
use crate::prob::pthmm::PitchHmm;
use crate::score::Score;

/// `DEFAULT_TEMPO` of the reference (BPM) when neither the user nor the score gives one.
pub const DEFAULT_TEMPO: f64 = 120.0;

/// A MIDI score-following method (reference name in parentheses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `pthmm`: pitch HMM with insertion states (MIDI default).
    Pthmm,
    /// `hmm`: pitch + inter-onset-interval HMM with a Kalman tempo model.
    Hmm,
    /// `outerhmm`: outer-product HMM following note by note with chord folding.
    OuterHmm,
}

impl Method {
    /// Reference names of the supported methods.
    pub const AVAILABLE: [&'static str; 3] = ["pthmm", "hmm", "outerhmm"];

    /// Reference name of the method.
    pub fn name(self) -> &'static str {
        match self {
            Method::Pthmm => "pthmm",
            Method::Hmm => "hmm",
            Method::OuterHmm => "outerhmm",
        }
    }
}

impl FromStr for Method {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "pthmm" => Ok(Method::Pthmm),
            "hmm" => Ok(Method::Hmm),
            "outerhmm" => Ok(Method::OuterHmm),
            _ => Err(Error::InvalidMethod {
                method: name.to_string(),
                available: Method::AVAILABLE.iter().map(|s| s.to_string()).collect(),
            }),
        }
    }
}

/// Score follower pipeline for one score.
pub struct Matchmaker {
    score: Score,
    tempo: f64,
    polling_period: f64,
    processor: Box<dyn Processor>,
    follower: Box<dyn OnlineAlignment>,
    /// Messages of the window that is still open (push mode).
    window_messages: Vec<MidiMessage>,
    window: usize,
    last_time: f64,
}

impl Matchmaker {
    /// Load the score and build the method's processor and follower. `tempo` overrides the
    /// score tempo; `unfold` is `unfold_score`.
    pub fn new(
        score_path: impl AsRef<Path>,
        method: Method,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        let score = Score::load(score_path, unfold)?;
        let tempo = tempo.or(score.tempo).unwrap_or(DEFAULT_TEMPO);
        let (polling_period, processor, follower): (
            f64,
            Box<dyn Processor>,
            Box<dyn OnlineAlignment>,
        ) = match method {
            Method::Pthmm => (
                0.01,
                Box::new(PitchProcessor::new(true, false)),
                Box::new(PitchHmm::new(&score, true, true)),
            ),
            Method::Hmm => (
                0.01,
                Box::new(PitchProcessor::new(true, true)),
                Box::new(PitchIoiHmm::new(&score, PitchIoiHmmConfig::default())),
            ),
            Method::OuterHmm => (
                0.001,
                Box::new(PitchProcessor::new(true, false)),
                Box::new(OuterProductHmm::new(
                    &score,
                    OuterProductHmmConfig::default(),
                )),
            ),
        };
        Ok(Matchmaker {
            score,
            tempo,
            polling_period,
            processor,
            follower,
            window_messages: Vec::new(),
            window: 0,
            last_time: f64::NEG_INFINITY,
        })
    }

    /// Simulate the performance MIDI file (`MidiStream` file mode) and return every position
    /// the follower yields.
    pub fn run_file(&mut self, performance_path: impl AsRef<Path>) -> Result<Vec<f64>> {
        let mut positions = Vec::new();
        for msg in Performance::load(performance_path)?.messages() {
            positions.extend(self.push(msg)?);
        }
        positions.extend(self.finish());
        Ok(positions)
    }

    /// Feed one message of a live source (FR-005); returns the positions of the windows it
    /// closes. Times must not decrease.
    pub fn push(&mut self, msg: MidiMessage) -> Result<Vec<f64>> {
        if msg.time < self.last_time {
            return Err(Error::InvalidParameter {
                key: "time".into(),
                reason: format!("{} after {}", msg.time, self.last_time),
            });
        }
        self.last_time = msg.time;
        let pp = self.polling_period;
        let mut positions = Vec::new();
        // same comparisons as midi_messages_to_framed_midi: window k holds k*pp < t <= (k+1)*pp
        while msg.time > (self.window + 1) as f64 * pp {
            positions.extend(self.close_window());
        }
        self.window_messages.push(msg);
        Ok(positions)
    }

    /// End of stream: close the remaining windows (`ceil(last_time / pp)` in total) and flush
    /// the processor.
    pub fn finish(&mut self) -> Vec<f64> {
        let n_frames = (self.last_time / self.polling_period).ceil();
        let mut positions = Vec::new();
        while (self.window as f64) < n_frames {
            positions.extend(self.close_window());
        }
        if let Some(obs) = self.processor.flush_remaining() {
            positions.extend(self.follow(&obs));
        }
        positions
    }

    fn close_window(&mut self) -> Option<f64> {
        let pp = self.polling_period;
        let frame = Frame {
            messages: std::mem::take(&mut self.window_messages),
            time: self.window as f64 * pp + 0.5 * pp,
        };
        self.window += 1;
        let obs = self.processor.process(&frame)?;
        self.follow(&obs)
    }

    fn follow(&mut self, obs: &crate::features::Observation) -> Option<f64> {
        self.follower
            .is_still_following()
            .then(|| self.follower.call(obs))
    }

    /// Replace processor and follower by wrappers around them, e.g. timing decorators of a
    /// benchmark harness (the library itself never reads a clock, research R9).
    pub fn wrap_stages(
        mut self,
        wrap: impl FnOnce(
            Box<dyn Processor>,
            Box<dyn OnlineAlignment>,
        ) -> (Box<dyn Processor>, Box<dyn OnlineAlignment>),
    ) -> Self {
        (self.processor, self.follower) = wrap(self.processor, self.follower);
        self
    }

    /// The loaded score.
    pub fn score(&self) -> &Score {
        &self.score
    }

    /// Tempo in BPM: user value, else score marking, else 120.
    pub fn tempo(&self) -> f64 {
        self.tempo
    }

    /// Alignment path so far.
    pub fn alignment_path(&self) -> &AlignmentPath {
        self.follower.alignment_path()
    }
}
