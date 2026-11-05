//! Port of `matchmaker.matchmaker.Matchmaker` for MIDI input: score + stream + processor +
//! follower, with the per-method defaults of the reference's `methods.yaml` (research R1).
//!
//! Upstream: `matchmaker/matchmaker.py`, `matchmaker/methods.yaml`.
//!
//! ```
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?;
//! let positions = mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;
//! println!("{} positions, path of {}", positions.len(), mm.alignment_path().perf_time.len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::dp::Distance;
use crate::dp::arzt::{ArztEvent, ArztEventConfig};
use crate::dp::arzt_frame::ArztFrame;
use crate::dp::arzt_multi_fold::{ArztMultiFoldCandidate, ArztMultiFoldFrame};
use crate::dp::arzt_tempo::ArztTempoFrame;
use crate::dp::dixon::{DixonEvent, DixonEventConfig};
use crate::dp::dixon_frame::DixonFrame;
use crate::error::{Error, Result};
use crate::features::{
    AudioProcessor, Observation, Processor, chord_onset::ChordOnsetProcessor,
    chroma::ChromaProcessor, lse::LseProcessor, pitch::PitchProcessor,
};
use crate::io::audio::AudioStream;
use crate::io::stream::{Frame, MidiMessage};
use crate::performance::Performance;
use crate::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
use crate::prob::outerhmm::{OuterProductHmm, OuterProductHmmConfig};
use crate::prob::pthmm::PitchHmm;
use crate::score::Score;
use crate::score::audio::{ScoreAudioReference, build_reference};

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
    /// `arzt`: event-level online time warping over the onset pianoroll.
    Arzt,
    /// `dixon`: event-level online time warping with a sparse, path-normalised cost matrix.
    Dixon,
}

impl Method {
    /// Reference names of the supported methods.
    pub const AVAILABLE: [&'static str; 5] = ["pthmm", "hmm", "outerhmm", "arzt", "dixon"];

    /// Keys [`MethodConfig::set`] accepts for this method.
    pub fn config_keys(self) -> &'static [&'static str] {
        match self {
            Method::OuterHmm => &[
                "processor",
                "piano_range",
                "polling_period",
                "patience",
                "other_prob",
            ],
            Method::Arzt => &[
                "processor",
                "piano_range",
                "polling_period",
                "step_size",
                "distance_func",
            ],
            Method::Dixon => &[
                "processor",
                "piano_range",
                "polling_period",
                "window_size",
                "max_run_count",
                "distance_func",
            ],
            _ => &["processor", "piano_range", "polling_period"],
        }
    }

    /// Reference name of the method.
    pub fn name(self) -> &'static str {
        match self {
            Method::Pthmm => "pthmm",
            Method::Hmm => "hmm",
            Method::OuterHmm => "outerhmm",
            Method::Arzt => "arzt",
            Method::Dixon => "dixon",
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
            "arzt" => Ok(Method::Arzt),
            "dixon" => Ok(Method::Dixon),
            _ => Err(Error::InvalidMethod {
                method: name.to_string(),
                available: Method::AVAILABLE.iter().map(|s| s.to_string()).collect(),
            }),
        }
    }
}

/// A frame-level score-following method for prerecorded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMethod {
    /// Frame-level OLTW with a bounded step advance.
    Arzt,
    /// Frame-level OLTW with a normalized sparse cost search.
    Dixon,
    /// Seeded tempo-aware frame-level Arzt OLTW.
    ArztTempo,
    /// Parallel frame-level Arzt OLTW over repeat/unfolding hypotheses.
    ArztMultiFold,
}

impl AudioMethod {
    /// Reference names of the supported audio methods.
    pub const AVAILABLE: [&'static str; 4] = ["arzt", "dixon", "arzt_tempo", "arzt_multi_fold"];

    /// Reference name of this method.
    pub fn name(self) -> &'static str {
        match self {
            AudioMethod::Arzt => "arzt",
            AudioMethod::Dixon => "dixon",
            AudioMethod::ArztTempo => "arzt_tempo",
            AudioMethod::ArztMultiFold => "arzt_multi_fold",
        }
    }
}

impl FromStr for AudioMethod {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "arzt" => Ok(AudioMethod::Arzt),
            "dixon" => Ok(AudioMethod::Dixon),
            "arzt_tempo" => Ok(AudioMethod::ArztTempo),
            "arzt_multi_fold" => Ok(AudioMethod::ArztMultiFold),
            _ => Err(Error::AudioConfig {
                key: "method".into(),
                reason: format!(
                    "unknown audio method {name:?}; available: {:?}",
                    AudioMethod::AVAILABLE
                ),
            }),
        }
    }
}

/// Feature processor (processor keys of the reference's `methods.yaml`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorKind {
    /// `pitch`: pitch vector, or pitch list for `hmm`.
    Pitch,
    /// `chord_onset`: one binary vector per chord.
    ChordOnset,
}

impl ProcessorKind {
    /// Reference names of the supported processors.
    pub const AVAILABLE: [&'static str; 2] = ["pitch", "chord_onset"];

    /// Reference name of the processor.
    pub fn name(self) -> &'static str {
        match self {
            ProcessorKind::Pitch => "pitch",
            ProcessorKind::ChordOnset => "chord_onset",
        }
    }
}

impl FromStr for ProcessorKind {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "pitch" => Ok(ProcessorKind::Pitch),
            "chord_onset" => Ok(ProcessorKind::ChordOnset),
            _ => Err(Error::InvalidParameter {
                key: "processor".into(),
                reason: format!(
                    "unknown processor {name:?}; available: {:?}",
                    ProcessorKind::AVAILABLE
                ),
            }),
        }
    }
}

/// Feature processor used by a frame-level audio method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioProcessorKind {
    /// Twelve-bin chromagram from an STFT.
    Chroma,
    /// Log-spectral energy onset feature.
    Lse,
}

impl AudioProcessorKind {
    /// Reference names of the supported audio processors.
    pub const AVAILABLE: [&'static str; 2] = ["chroma", "lse"];

    /// Reference name of this processor.
    pub fn name(self) -> &'static str {
        match self {
            AudioProcessorKind::Chroma => "chroma",
            AudioProcessorKind::Lse => "lse",
        }
    }
}

impl FromStr for AudioProcessorKind {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "chroma" => Ok(AudioProcessorKind::Chroma),
            "lse" => Ok(AudioProcessorKind::Lse),
            _ => Err(Error::AudioConfig {
                key: "processor".into(),
                reason: format!(
                    "unknown audio processor {name:?}; available: {:?}",
                    AudioProcessorKind::AVAILABLE
                ),
            }),
        }
    }
}

/// Local distance functions used by audio-frame followers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDistance {
    /// Manhattan (L1) distance, the audio Arzt default.
    Manhattan,
    /// Euclidean (L2) distance, the audio Dixon default.
    Euclidean,
}

impl FromStr for AudioDistance {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "manhattan" => Ok(AudioDistance::Manhattan),
            "euclidean" => Ok(AudioDistance::Euclidean),
            _ => Err(Error::AudioConfig {
                key: "distance_func".into(),
                reason: format!(
                    "unknown audio distance {name:?}; available: [\"manhattan\", \"euclidean\"]"
                ),
            }),
        }
    }
}

/// Configuration kept separate from [`MethodConfig`] so audio defaults cannot change MIDI.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioMethodConfig {
    /// Feature processor.
    processor: AudioProcessorKind,
    /// Target mono sample rate in hertz.
    pub sample_rate: u32,
    /// Frame rate when `hop_length` is `None`.
    pub frame_rate: u32,
    /// Explicit hop length in samples; overrides `frame_rate` when set.
    pub hop_length: Option<usize>,
    /// Per-frame processor norm (`None` disables normalization; infinity is valid).
    pub norm: Option<f64>,
    /// Required local SF2 path for runtime score synthesis.
    pub soundfont_path: Option<std::path::PathBuf>,
    /// Search window in seconds.
    pub window_size: f64,
    /// Arzt initial search window in seconds.
    pub start_window_size: Option<f64>,
    /// Arzt maximum score-frame advance per input frame.
    pub step_size: Option<usize>,
    /// Dixon maximum consecutive one-sided advances.
    pub max_run_count: Option<usize>,
    /// Local feature distance.
    pub distance_func: AudioDistance,
    /// Explicit MT19937 seed for tempo-aware Arzt; invalid on deterministic frame methods.
    pub random_seed: Option<u32>,
    /// Number of recent frame costs used to select a repeat hypothesis.
    pub selection_window: Option<usize>,
    /// Relative improvement required to switch repeat hypotheses.
    pub switch_margin: Option<f64>,
}

impl AudioMethodConfig {
    /// Reference defaults for the selected audio method.
    pub fn for_method(method: AudioMethod) -> Self {
        let arzt = matches!(
            method,
            AudioMethod::Arzt | AudioMethod::ArztTempo | AudioMethod::ArztMultiFold
        );
        let processor = if arzt {
            AudioProcessorKind::Chroma
        } else {
            AudioProcessorKind::Lse
        };
        Self {
            processor,
            sample_rate: 44_100,
            frame_rate: if arzt { 30 } else { 50 },
            hop_length: None,
            norm: Some(processor.default_norm()),
            soundfont_path: None,
            window_size: 10.0,
            start_window_size: arzt.then_some(0.1),
            step_size: arzt.then_some(3),
            max_run_count: (!arzt).then_some(3),
            distance_func: if arzt {
                AudioDistance::Manhattan
            } else {
                AudioDistance::Euclidean
            },
            random_seed: (method == AudioMethod::ArztTempo).then_some(1984),
            selection_window: (method == AudioMethod::ArztMultiFold).then_some(50),
            switch_margin: (method == AudioMethod::ArztMultiFold).then_some(0.05),
        }
    }

    /// Selected feature processor.
    pub fn processor(&self) -> AudioProcessorKind {
        self.processor
    }

    /// Select a processor and restore its reference normalization default.
    /// Set `norm` afterward to provide an explicit override.
    pub fn set_processor(&mut self, processor: AudioProcessorKind) {
        self.processor = processor;
        self.norm = Some(processor.default_norm());
    }

    /// Validate shared audio values and method-specific parameter ranges.
    pub fn validate(&self, method: AudioMethod) -> Result<()> {
        let invalid = |key: &str, reason: &str| Error::AudioConfig {
            key: key.into(),
            reason: reason.into(),
        };
        if self.sample_rate == 0 {
            return Err(invalid("sample_rate", "must be at least 1 Hz"));
        }
        if self.hop_length == Some(0) {
            return Err(invalid("hop_length", "must be at least 1 sample"));
        }
        if self.hop_length.is_none() && self.frame_rate == 0 {
            return Err(invalid("frame_rate", "must be at least 1 frame per second"));
        }
        if !self.window_size.is_finite() || self.window_size <= 0.0 {
            return Err(invalid(
                "window_size",
                "must be a positive number of seconds",
            ));
        }
        if self.norm.is_some_and(|norm| norm.is_nan() || norm <= 0.0) {
            return Err(invalid("norm", "must be positive or None"));
        }
        if self.random_seed.is_some() && method != AudioMethod::ArztTempo {
            return Err(invalid("random_seed", "is only valid for audio arzt_tempo"));
        }
        if self.selection_window.is_some() && method != AudioMethod::ArztMultiFold {
            return Err(invalid(
                "selection_window",
                "is only valid for audio arzt_multi_fold",
            ));
        }
        if self.switch_margin.is_some() && method != AudioMethod::ArztMultiFold {
            return Err(invalid(
                "switch_margin",
                "is only valid for audio arzt_multi_fold",
            ));
        }
        if self.selection_window == Some(0) {
            return Err(invalid("selection_window", "must be at least 1"));
        }
        if self
            .switch_margin
            .is_some_and(|margin| !margin.is_finite() || !(0.0..=1.0).contains(&margin))
        {
            return Err(invalid(
                "switch_margin",
                "must be finite and between 0 and 1",
            ));
        }
        if method == AudioMethod::ArztMultiFold
            && (self.selection_window.is_none() || self.switch_margin.is_none())
        {
            return Err(invalid(
                "selection_window/switch_margin",
                "are required for audio arzt_multi_fold",
            ));
        }
        match method {
            AudioMethod::Arzt | AudioMethod::ArztTempo | AudioMethod::ArztMultiFold => {
                if self.max_run_count.is_some() {
                    return Err(invalid("max_run_count", "is only valid for audio dixon"));
                }
                if self
                    .start_window_size
                    .is_some_and(|size| !size.is_finite() || size <= 0.0)
                {
                    return Err(invalid(
                        "start_window_size",
                        "must be a positive number of seconds",
                    ));
                }
                if self.step_size == Some(0) {
                    return Err(invalid("step_size", "must be at least 1"));
                }
            }
            AudioMethod::Dixon => {
                if self.start_window_size.is_some() || self.step_size.is_some() {
                    return Err(invalid(
                        "start_window_size/step_size",
                        "are only valid for audio arzt",
                    ));
                }
                if self.max_run_count == Some(0) {
                    return Err(invalid("max_run_count", "must be at least 1"));
                }
            }
        }
        Ok(())
    }
}

impl AudioProcessorKind {
    fn default_norm(self) -> f64 {
        match self {
            AudioProcessorKind::Chroma => f64::INFINITY,
            AudioProcessorKind::Lse => 2.0,
        }
    }
}

/// A value of a `Matchmaker(kwargs=...)` entry.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// Python `None`.
    None,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
}

/// What `Matchmaker(kwargs=...)` controls: processor, `piano_range`, the stream's
/// `polling_period` (`None` = event-based, one message per frame) and the parameters of
/// `outerhmm`, `arzt` and `dixon`.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodConfig {
    /// Feature processor.
    pub processor: ProcessorKind,
    /// Pitches are restricted to the 88 piano keys.
    pub piano_range: bool,
    /// Window length in seconds; `None` streams one message per frame.
    pub polling_period: Option<f64>,
    /// `outerhmm`: chords without a state change before following stops (default 10).
    pub patience: Option<usize>,
    /// `outerhmm`: probability of pitches outside every error class (default 1e-6).
    pub other_prob: Option<f64>,
    /// `arzt`: largest advance of the position per observation (default 5).
    pub step_size: Option<usize>,
    /// `dixon`: search width in observations (default 30).
    pub window_size: Option<usize>,
    /// `dixon`: consecutive advances in one direction before the other is forced (default 3).
    pub max_run_count: Option<usize>,
    /// `arzt`, `dixon`: local cost (default `cosine` / `euclidean`).
    pub distance_func: Option<Distance>,
}

impl MethodConfig {
    /// The `default_kwargs` of `method`: piano range, `pitch` (`arzt`, `dixon`: `chord_onset`) and
    /// windows of 10 ms (`outerhmm`, `arzt`, `dixon`: 1 ms).
    pub fn for_method(method: Method) -> Self {
        let oltw = matches!(method, Method::Arzt | Method::Dixon);
        MethodConfig {
            processor: if oltw {
                ProcessorKind::ChordOnset
            } else {
                ProcessorKind::Pitch
            },
            piano_range: true,
            polling_period: Some(if oltw || method == Method::OuterHmm {
                0.001
            } else {
                0.01
            }),
            patience: None,
            other_prob: None,
            step_size: None,
            window_size: None,
            max_run_count: None,
            distance_func: None,
        }
    }

    /// Set one `kwargs` entry; unknown keys and invalid values are [`Error::InvalidParameter`].
    pub fn set(&mut self, method: Method, key: &str, value: &ConfigValue) -> Result<()> {
        let invalid = |reason: String| Error::InvalidParameter {
            key: key.to_string(),
            reason,
        };
        if !method.config_keys().contains(&key) {
            if method == Method::Arzt && matches!(key, "window_size" | "start_window_size") {
                return Err(invalid("not used by the reference's MIDI arzt".into()));
            }
            return Err(invalid(format!(
                "unknown parameter for {}; allowed: {:?}",
                method.name(),
                method.config_keys()
            )));
        }
        match (key, value) {
            ("processor", ConfigValue::Str(name)) => self.processor = name.parse()?,
            ("piano_range", ConfigValue::Bool(b)) => self.piano_range = *b,
            ("polling_period", ConfigValue::None) => self.polling_period = None,
            ("polling_period", ConfigValue::Float(p)) => self.polling_period = Some(*p),
            ("polling_period", ConfigValue::Int(p)) => self.polling_period = Some(*p as f64),
            ("patience", ConfigValue::Int(p)) => {
                self.patience =
                    Some(usize::try_from(*p).map_err(|_| invalid("must not be negative".into()))?);
            }
            ("other_prob", ConfigValue::Float(p)) => self.other_prob = Some(*p),
            ("other_prob", ConfigValue::Int(p)) => self.other_prob = Some(*p as f64),
            ("step_size" | "window_size" | "max_run_count", ConfigValue::Int(n)) => {
                let n = usize::try_from(*n).map_err(|_| invalid("must not be negative".into()))?;
                match key {
                    "step_size" => self.step_size = Some(n),
                    "window_size" => self.window_size = Some(n),
                    _ => self.max_run_count = Some(n),
                }
            }
            ("distance_func", ConfigValue::Str(name)) => self.distance_func = Some(name.parse()?),
            _ => return Err(invalid(format!("unsupported value {value:?}"))),
        }
        Ok(())
    }

    /// Reject combinations the method cannot run with.
    fn validate(&self, method: Method) -> Result<()> {
        let invalid = |key: &str, reason: &str| Error::InvalidParameter {
            key: key.into(),
            reason: reason.into(),
        };
        if self
            .polling_period
            .is_some_and(|p| !(p.is_finite() && p > 0.0))
        {
            return Err(invalid(
                "polling_period",
                "must be a positive number of seconds or None",
            ));
        }
        if method == Method::Hmm && self.processor != ProcessorKind::Pitch {
            return Err(invalid(
                "processor",
                "hmm needs the pitch processor (pitch list)",
            ));
        }
        if method == Method::OuterHmm && !self.piano_range {
            return Err(invalid(
                "piano_range",
                "outerhmm needs the 88-key piano range",
            ));
        }
        for (key, size) in [
            ("step_size", self.step_size),
            ("window_size", self.window_size),
            ("max_run_count", self.max_run_count),
        ] {
            if size == Some(0) {
                return Err(invalid(key, "must be at least 1"));
            }
        }
        Ok(())
    }
}

/// One follower call of a recorded run.
#[derive(Debug, Clone)]
pub struct Step {
    /// What the follower received.
    pub observation: Observation,
    /// The position it returned.
    pub position: f64,
    /// Its state afterwards.
    pub debug: StepDebug,
}

/// Intermediate data of a run, kept after [`Matchmaker::record`] (input of [`crate::export`]).
#[derive(Debug, Clone, Default)]
pub struct Recording {
    /// Every stream frame, in order (empty windows included).
    pub frames: Vec<Frame>,
    /// Every processor output, also those after the follower stopped.
    pub processor_outputs: Vec<Observation>,
    /// Every follower call.
    pub steps: Vec<Step>,
}

/// Score follower pipeline for one score.
pub struct Matchmaker {
    score: Score,
    method: Method,
    config: MethodConfig,
    tempo: f64,
    processor: Box<dyn Processor>,
    follower: Box<dyn OnlineAlignment>,
    recording: Option<Recording>,
    /// Messages of the window that is still open (push mode).
    window_messages: Vec<MidiMessage>,
    window: usize,
    last_time: f64,
}

impl Matchmaker {
    /// Load the score and build the method's processor and follower with the method's default
    /// configuration. `tempo` overrides the score tempo; `unfold` is `unfold_score`.
    pub fn new(
        score_path: impl AsRef<Path>,
        method: Method,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        Self::with_config(
            score_path,
            method,
            MethodConfig::for_method(method),
            tempo,
            unfold,
        )
    }

    /// Like [`Matchmaker::new`] with an explicit [`MethodConfig`].
    pub fn with_config(
        score_path: impl AsRef<Path>,
        method: Method,
        config: MethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        config.validate(method)?;
        let score = Score::load(score_path, unfold)?;
        let tempo = tempo.or(score.tempo).unwrap_or(DEFAULT_TEMPO);
        let processor: Box<dyn Processor> = match config.processor {
            ProcessorKind::Pitch => Box::new(PitchProcessor::new(
                config.piano_range,
                method == Method::Hmm,
            )),
            ProcessorKind::ChordOnset => Box::new(ChordOnsetProcessor::new(config.piano_range)),
        };
        let follower: Box<dyn OnlineAlignment> = match method {
            Method::Pthmm => Box::new(PitchHmm::new(&score, true, config.piano_range)),
            Method::Hmm => Box::new(PitchIoiHmm::new(
                &score,
                PitchIoiHmmConfig {
                    piano_range: config.piano_range,
                    ..PitchIoiHmmConfig::default()
                },
            )),
            Method::OuterHmm => {
                let defaults = OuterProductHmmConfig::default();
                Box::new(OuterProductHmm::new(
                    &score,
                    OuterProductHmmConfig {
                        patience: config.patience.unwrap_or(defaults.patience),
                        other_prob: config.other_prob.unwrap_or(defaults.other_prob),
                        ..defaults
                    },
                ))
            }
            Method::Arzt => {
                let (features, onsets) = score.onset_pianoroll(config.piano_range);
                let defaults = ArztEventConfig::default();
                Box::new(ArztEvent::new(
                    features,
                    onsets,
                    ArztEventConfig {
                        step_size: config.step_size.unwrap_or(defaults.step_size),
                        distance: config.distance_func.unwrap_or(defaults.distance),
                        ..defaults
                    },
                ))
            }
            Method::Dixon => {
                let (features, onsets) = score.onset_pianoroll(config.piano_range);
                let defaults = DixonEventConfig::default();
                Box::new(DixonEvent::new(
                    features,
                    onsets,
                    DixonEventConfig {
                        window_size: config.window_size.unwrap_or(defaults.window_size),
                        max_run_count: config.max_run_count.unwrap_or(defaults.max_run_count),
                        distance: config.distance_func.unwrap_or(defaults.distance),
                    },
                ))
            }
        };
        Ok(Matchmaker {
            score,
            method,
            config,
            tempo,
            processor,
            follower,
            recording: None,
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
        let Some(pp) = self.config.polling_period else {
            // event-based: every message is its own frame at its own time
            let frame = Frame {
                time: msg.time,
                messages: vec![msg],
            };
            return Ok(self.process_frame(&frame).into_iter().collect());
        };
        let mut positions = Vec::new();
        // same comparisons as midi_messages_to_framed_midi: window k holds k*pp < t <= (k+1)*pp
        while msg.time > (self.window + 1) as f64 * pp {
            positions.extend(self.close_window(pp));
        }
        self.window_messages.push(msg);
        Ok(positions)
    }

    /// End of stream: close the remaining windows (`ceil(last_time / pp)` in total) and flush
    /// the processor.
    pub fn finish(&mut self) -> Vec<f64> {
        let mut positions = Vec::new();
        if let Some(pp) = self.config.polling_period {
            let n_frames = (self.last_time / pp).ceil();
            while (self.window as f64) < n_frames {
                positions.extend(self.close_window(pp));
            }
        }
        if let Some(obs) = self.processor.flush_remaining() {
            positions.extend(self.follow(&obs));
        }
        positions
    }

    fn close_window(&mut self, pp: f64) -> Option<f64> {
        let frame = Frame {
            messages: std::mem::take(&mut self.window_messages),
            time: self.window as f64 * pp + 0.5 * pp,
        };
        self.window += 1;
        self.process_frame(&frame)
    }

    fn process_frame(&mut self, frame: &Frame) -> Option<f64> {
        let _span = tracing::trace_span!("stream").entered();
        if let Some(recording) = &mut self.recording {
            recording.frames.push(frame.clone());
        }
        let obs = self.processor.process(frame)?;
        self.follow(&obs)
    }

    fn follow(&mut self, obs: &Observation) -> Option<f64> {
        if let Some(recording) = &mut self.recording {
            recording.processor_outputs.push(obs.clone());
        }
        if !self.follower.is_still_following() {
            return None;
        }
        let position = self.follower.call(obs);
        if let Some(recording) = &mut self.recording {
            recording.steps.push(Step {
                observation: obs.clone(),
                position,
                debug: self.follower.debug_state(),
            });
        }
        Some(position)
    }

    /// Keep the intermediate data of the run (memory grows with the performance; off by default).
    pub fn record(mut self) -> Self {
        self.recording = Some(Recording::default());
        self
    }

    /// The recorded run, if [`Matchmaker::record`] was called.
    pub fn recording(&self) -> Option<&Recording> {
        self.recording.as_ref()
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

    /// The method this pipeline runs.
    pub fn method(&self) -> Method {
        self.method
    }

    /// The effective configuration.
    pub fn config(&self) -> &MethodConfig {
        &self.config
    }

    /// Window length of the stream in seconds; `None` for event-based streaming.
    pub fn polling_period(&self) -> Option<f64> {
        self.config.polling_period
    }

    /// Ascending score beat of every note onset (`Matchmaker.score_positions`).
    pub fn score_positions(&self) -> Vec<f32> {
        self.score.unique_onsets()
    }

    /// States of the follower in beats (with inserted states for `pthmm` and `hmm`).
    pub fn follower_score_positions(&self) -> &[f32] {
        self.follower.score_positions()
    }
}

/// Core pipeline for following prerecorded audio with frame-level `arzt`.
pub struct AudioMatchmaker {
    method: AudioMethod,
    score: Score,
    config: AudioMethodConfig,
    tempo: f64,
    reference: ScoreAudioReference,
    multi_fold_candidates: Vec<ArztMultiFoldCandidate>,
    processor: Box<dyn AudioProcessor>,
    cache_size: usize,
    follower: Box<dyn OnlineAlignment>,
    recording: Option<AudioRecording>,
}

/// Intermediate audio stages retained by [`AudioMatchmaker::record`].
#[derive(Debug, Clone, Default)]
pub struct AudioRecording {
    /// Path of the decoded performance file.
    pub performance_path: Option<PathBuf>,
    /// Input sample rate before resampling.
    pub source_sample_rate: u32,
    /// Input channel count before mono conversion.
    pub source_channels: usize,
    /// Time of each emitted audio frame.
    pub frame_times: Vec<f64>,
    /// Processor outputs for each emitted audio frame.
    pub processor_outputs: Vec<Observation>,
    /// Time spent extracting each yielded performance feature frame.
    pub feature_durations: Vec<Duration>,
    /// Time spent on each follower inference step.
    pub inference_durations: Vec<Duration>,
    /// Follower calls before the score stop condition.
    pub steps: Vec<Step>,
}

impl AudioMatchmaker {
    /// Load a score and build its audio reference for the audio `arzt` method.
    pub fn new_arzt(
        score_path: impl AsRef<Path>,
        config: AudioMethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        Self::new_audio(score_path, AudioMethod::Arzt, config, tempo, unfold)
    }

    /// Load a score and build its audio reference for seeded tempo-aware Arzt.
    pub fn new_arzt_tempo(
        score_path: impl AsRef<Path>,
        config: AudioMethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        Self::new_audio(score_path, AudioMethod::ArztTempo, config, tempo, unfold)
    }

    /// Load a score and build parallel repeat hypotheses for audio `arzt_multi_fold`.
    pub fn new_arzt_multi_fold(
        score_path: impl AsRef<Path>,
        config: AudioMethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        Self::new_audio(
            score_path,
            AudioMethod::ArztMultiFold,
            config,
            tempo,
            unfold,
        )
    }

    /// Load a score and build its audio reference for the audio `dixon` method.
    pub fn new_dixon(
        score_path: impl AsRef<Path>,
        config: AudioMethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        Self::new_audio(score_path, AudioMethod::Dixon, config, tempo, unfold)
    }

    fn new_audio(
        score_path: impl AsRef<Path>,
        method: AudioMethod,
        config: AudioMethodConfig,
        tempo: Option<f64>,
        unfold: bool,
    ) -> Result<Self> {
        config.validate(method)?;
        let soundfont_path = config
            .soundfont_path
            .as_ref()
            .ok_or_else(|| Error::AudioConfig {
                key: "soundfont_path".into(),
                reason: "is required for score rendering".into(),
            })?;
        let hop_length = config
            .hop_length
            .unwrap_or((config.sample_rate / config.frame_rate) as usize);
        if hop_length == 0 {
            return Err(Error::AudioConfig {
                key: "frame_rate".into(),
                reason: "must not exceed sample_rate".into(),
            });
        }

        let mut score = Score::load(score_path, unfold)?;
        if let Some(tempo) = tempo {
            score.tempo = Some(tempo);
        }
        let tempo = score.tempo.unwrap_or(DEFAULT_TEMPO);
        let (reference, follower, multi_fold_candidates) = if method == AudioMethod::ArztMultiFold {
            let mut candidates = Vec::new();
            let mut references = Vec::new();
            for (index, candidate_score) in score.multi_fold_variants()?.into_iter().enumerate() {
                let reference = build_reference(&candidate_score, soundfont_path, &config, method)?;
                let max_onset_beat = candidate_score
                    .notes
                    .iter()
                    .map(|note| note.onset_beat)
                    .fold(f32::NEG_INFINITY, f32::max);
                candidates.push(ArztMultiFoldCandidate {
                    index,
                    note_count: candidate_score.notes.len(),
                    max_onset_beat,
                    features: reference.features.clone(),
                    score_positions: reference.score_positions.clone(),
                    unfolded_frame_to_beat: reference.frame_to_beat.clone(),
                    notated_frame_to_beat: multi_fold_frame_to_notated(
                        &score,
                        &candidate_score,
                        &reference.frame_to_beat,
                    ),
                });
                references.push(reference);
            }
            let reference = references.remove(0);
            let follower = Box::new(ArztMultiFoldFrame::new(
                candidates.clone(),
                score.unique_onsets(),
                config.clone(),
            )?);
            (reference, follower as Box<dyn OnlineAlignment>, candidates)
        } else {
            let reference = build_reference(&score, soundfont_path, &config, method)?;
            let follower: Box<dyn OnlineAlignment> = match method {
                AudioMethod::Arzt => Box::new(ArztFrame::new(
                    reference.features.clone(),
                    reference.score_positions.clone(),
                    reference.frame_to_beat.clone(),
                    config.clone(),
                )?),
                AudioMethod::Dixon => Box::new(DixonFrame::new(
                    reference.features.clone(),
                    reference.score_positions.clone(),
                    reference.frame_to_beat.clone(),
                    config.clone(),
                )?),
                AudioMethod::ArztTempo => Box::new(ArztTempoFrame::new(
                    reference.features.clone(),
                    reference.score_positions.clone(),
                    reference.frame_to_beat.clone(),
                    config.clone(),
                )?),
                AudioMethod::ArztMultiFold => unreachable!("handled above"),
            };
            (reference, follower, Vec::new())
        };
        let (processor, cache_size) = make_audio_processor(&config, hop_length);

        Ok(Self {
            method,
            score,
            config,
            tempo,
            reference,
            multi_fold_candidates,
            processor,
            cache_size,
            follower,
            recording: None,
        })
    }

    /// Retain the intermediate stages for [`crate::export::write_audio_run`].
    pub fn record(mut self) -> Self {
        self.recording = Some(AudioRecording::default());
        self
    }

    /// Recorded stage data, if [`AudioMatchmaker::record`] was called.
    pub fn recording(&self) -> Option<&AudioRecording> {
        self.recording.as_ref()
    }

    /// Decode and follow one prerecorded audio file, returning one beat position per consumed frame.
    pub fn run_file(&mut self, performance_path: impl AsRef<Path>) -> Result<Vec<f64>> {
        let performance_path = performance_path.as_ref();
        let mut stream = AudioStream::open(
            performance_path,
            self.config.sample_rate,
            self.hop_length(),
            self.cache_size,
        )?;
        if let Some(recording) = &mut self.recording {
            recording.performance_path = Some(performance_path.to_path_buf());
            recording.source_sample_rate = stream.source_sample_rate();
            recording.source_channels = stream.source_channels();
        }
        stream.prime(self.processor.as_mut());
        let mut positions = Vec::new();
        for frame in stream {
            if !self.follower.is_still_following() {
                break;
            }
            if let Some(recording) = &mut self.recording {
                recording.frame_times.push(frame.perf_time);
            }
            let observation = if self.recording.is_some() {
                let start = Instant::now();
                let observation = self.processor.process(&frame);
                if observation.is_some()
                    && let Some(recording) = &mut self.recording
                {
                    recording.feature_durations.push(start.elapsed());
                }
                observation
            } else {
                self.processor.process(&frame)
            };
            if let Some(observation) = observation {
                if let Some(recording) = &mut self.recording {
                    recording.processor_outputs.push(observation.clone());
                }
                let inference_start = self.recording.as_ref().map(|_| Instant::now());
                let position = self.follower.call(&observation);
                positions.push(position);
                if let Some(recording) = &mut self.recording {
                    if let Some(start) = inference_start {
                        recording.inference_durations.push(start.elapsed());
                    }
                    recording.steps.push(Step {
                        observation,
                        position,
                        debug: self.follower.debug_state(),
                    });
                }
            }
        }
        Ok(positions)
    }

    /// The loaded score.
    pub fn score(&self) -> &Score {
        &self.score
    }

    /// Effective audio configuration.
    pub fn config(&self) -> &AudioMethodConfig {
        &self.config
    }

    /// The frame-level audio method used by this run.
    pub fn method(&self) -> AudioMethod {
        self.method
    }

    /// Effective score tempo in quarter-note BPM.
    pub fn tempo(&self) -> f64 {
        self.tempo
    }

    /// Synthesized reference waveform, features and per-frame beat map.
    pub fn reference(&self) -> &ScoreAudioReference {
        &self.reference
    }

    /// Ordered candidate references for multi-fold Arzt, empty for other methods.
    pub fn multi_fold_candidates(&self) -> &[ArztMultiFoldCandidate] {
        &self.multi_fold_candidates
    }

    /// Alignment path so far.
    pub fn alignment_path(&self) -> &AlignmentPath {
        self.follower.alignment_path()
    }

    /// Score onsets used to stop the follower.
    pub fn score_positions(&self) -> &[f32] {
        self.follower.score_positions()
    }

    fn hop_length(&self) -> usize {
        self.config
            .hop_length
            .unwrap_or((self.config.sample_rate / self.config.frame_rate) as usize)
    }
}

fn multi_fold_frame_to_notated(
    input_score: &Score,
    candidate_score: &Score,
    frame_to_unfolded_beat: &[f64],
) -> Vec<f64> {
    let folded_beats: HashMap<&str, f64> = input_score
        .notes
        .iter()
        .map(|note| (note.id.as_str(), f64::from(note.onset_beat)))
        .collect();
    let mut knots: Vec<(f64, f64)> = candidate_score
        .notes
        .iter()
        .filter_map(|note| {
            let id = strip_repeat_suffix(&note.id);
            folded_beats
                .get(id)
                .map(|&notated| (f64::from(note.onset_beat), notated))
        })
        .collect();
    knots.sort_by(|left, right| left.0.total_cmp(&right.0));
    knots.dedup_by(|left, right| left.0 == right.0);
    if knots.is_empty() {
        return frame_to_unfolded_beat
            .iter()
            .map(|beat| f64::from(*beat as f32))
            .collect();
    }
    frame_to_unfolded_beat
        .iter()
        .map(|&beat| {
            let beat = f64::from(beat as f32);
            let upper = knots.partition_point(|(unfolded, _)| *unfolded <= beat);
            if upper == 0 {
                return knots[0].1;
            }
            if upper == knots.len() {
                return knots[knots.len() - 1].1;
            }
            let (left_beat, left_notated) = knots[upper - 1];
            let (right_beat, right_notated) = knots[upper];
            left_notated
                + (right_notated - left_notated) * (beat - left_beat) / (right_beat - left_beat)
        })
        .collect()
}

fn strip_repeat_suffix(note_id: &str) -> &str {
    match note_id.rsplit_once('-') {
        Some((original, suffix))
            if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) =>
        {
            original
        }
        _ => note_id,
    }
}

fn make_audio_processor(
    config: &AudioMethodConfig,
    hop_length: usize,
) -> (Box<dyn AudioProcessor>, usize) {
    match config.processor {
        AudioProcessorKind::Chroma => {
            let processor = ChromaProcessor::new(config.sample_rate, hop_length, config.norm);
            let cache_size = processor.n_fft().saturating_sub(hop_length);
            (Box::new(processor), cache_size)
        }
        AudioProcessorKind::Lse => {
            let processor = LseProcessor::new(config.sample_rate, hop_length, config.norm);
            let cache_size = processor.n_fft().saturating_sub(hop_length);
            (Box::new(processor), cache_size)
        }
    }
}
