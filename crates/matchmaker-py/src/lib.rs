//! Python bindings of matchmaker-rs: a drop-in `Matchmaker` for pymatchmaker's MIDI usage.
//!
//! Ports the public surface of `matchmaker.matchmaker.Matchmaker` (MIDI methods only).
//! See `specs/001-midi-score-following/contracts/python-api.md`.

use std::collections::VecDeque;
use std::path::PathBuf;

use matchmaker_rs::Error;
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::export::write_run;
use matchmaker_rs::io::stream::MidiMessage;
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind, ConfigValue,
    Matchmaker as Core, Method, MethodConfig,
};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::score::{AnnotationLevel, Score};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{
    PyNotImplementedError, PyOSError, PyRuntimeError, PyStopIteration, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

/// Reference tag the build was validated against (`tools/golden/pyproject.toml`).
const REFERENCE_VERSION: &str = "v0.4.0";

/// `AVAILABLE_METHODS["midi"]` formatted like a Python list.
fn available_methods() -> String {
    let names: Vec<String> = Method::AVAILABLE.iter().map(|m| format!("'{m}'")).collect();
    format!("[{}]", names.join(", "))
}

fn available_audio_methods() -> String {
    "['arzt']".into()
}

/// Errors about parameters are `ValueError`; a file that cannot be read is `OSError`.
fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Io { .. } | Error::AudioDecode { .. } | Error::AudioRender { .. } => {
            PyOSError::new_err(err.to_string())
        }
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// `(2, T)` array: row 0 performance seconds, row 1 score beats (`(2, 0)` before the first step).
fn path_array<'py>(py: Python<'py>, path: &AlignmentPath) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = path
        .perf_time
        .iter()
        .chain(&path.score_beat)
        .copied()
        .collect();
    Array2::from_shape_vec((2, path.perf_time.len()), flat)
        .expect("both rows have the same length")
        .into_pyarray(py)
}

fn config_value(value: &Bound<'_, PyAny>) -> PyResult<ConfigValue> {
    if value.is_none() {
        Ok(ConfigValue::None)
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(ConfigValue::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(ConfigValue::Int(i))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(ConfigValue::Float(f))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(ConfigValue::Str(s))
    } else {
        Err(PyValueError::new_err(format!(
            "unsupported kwargs value {}",
            value.repr()?
        )))
    }
}

/// `DEFAULT_KWARGS["midi"][method]` of the reference's `methods.yaml`.
fn default_kwargs<'py>(py: Python<'py>, method: Method) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    let oltw = matches!(method, Method::Arzt | Method::Dixon);
    kwargs.set_item("processor", if oltw { "chord_onset" } else { "pitch" })?;
    kwargs.set_item("piano_range", true)?;
    match method {
        Method::OuterHmm => kwargs.set_item("polling_period", 0.001)?,
        Method::Arzt => {
            kwargs.set_item("polling_period", 0.001)?;
            kwargs.set_item("window_size", 2)?;
            kwargs.set_item("start_window_size", 2)?;
            kwargs.set_item("step_size", 5)?;
        }
        Method::Dixon => {
            kwargs.set_item("polling_period", 0.001)?;
            kwargs.set_item("window_size", 30)?;
        }
        Method::Pthmm | Method::Hmm => {}
    }
    Ok(kwargs)
}

fn default_audio_kwargs<'py>(py: Python<'py>, method: AudioMethod) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    if method == AudioMethod::Arzt {
        kwargs.set_item("processor", "chroma")?;
        kwargs.set_item("sample_rate", 44_100)?;
        kwargs.set_item("frame_rate", 30)?;
        kwargs.set_item("window_size", 10.0)?;
        kwargs.set_item("start_window_size", 0.1)?;
        kwargs.set_item("step_size", 3)?;
        kwargs.set_item("distance_func", "manhattan")?;
    }
    Ok(kwargs)
}

fn set_audio_config(
    config: &mut AudioMethodConfig,
    method: AudioMethod,
    key: &str,
    value: ConfigValue,
) -> PyResult<()> {
    let number = |value: &ConfigValue| match value {
        ConfigValue::Int(value) => Ok(*value as f64),
        ConfigValue::Float(value) => Ok(*value),
        _ => Err(PyValueError::new_err(format!(
            "Invalid audio parameter {key}"
        ))),
    };
    let integer = |value: &ConfigValue| match value {
        ConfigValue::Int(value) => usize::try_from(*value)
            .map_err(|_| PyValueError::new_err(format!("Invalid audio parameter {key}"))),
        _ => Err(PyValueError::new_err(format!(
            "Invalid audio parameter {key}"
        ))),
    };
    match key {
        "sample_rate" => {
            config.sample_rate = u32::try_from(integer(&value)?)
                .map_err(|_| PyValueError::new_err("Invalid audio parameter sample_rate"))?;
        }
        "frame_rate" => {
            config.frame_rate = u32::try_from(integer(&value)?)
                .map_err(|_| PyValueError::new_err("Invalid audio parameter frame_rate"))?;
        }
        "hop_length" => {
            config.hop_length = match value {
                ConfigValue::None => None,
                _ => Some(integer(&value)?),
            };
        }
        "norm" => {
            config.norm = match value {
                ConfigValue::None => None,
                ConfigValue::Str(ref value)
                    if value.eq_ignore_ascii_case("inf")
                        || value.eq_ignore_ascii_case("infinity") =>
                {
                    Some(f64::INFINITY)
                }
                _ => Some(number(&value)?),
            };
        }
        "window_size" => config.window_size = number(&value)?,
        "start_window_size" => {
            config.start_window_size = match value {
                ConfigValue::None => None,
                _ => Some(number(&value)?),
            };
        }
        "step_size" if method == AudioMethod::Arzt => {
            config.step_size = match value {
                ConfigValue::None => None,
                _ => Some(integer(&value)?),
            };
        }
        "max_run_count" => {
            config.max_run_count = match value {
                ConfigValue::None => None,
                _ => Some(integer(&value)?),
            };
        }
        "distance_func" => {
            let ConfigValue::Str(value) = value else {
                return Err(PyValueError::new_err(
                    "Invalid audio parameter distance_func",
                ));
            };
            config.distance_func = value.parse().map_err(to_py_err)?;
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid audio parameter {key}"
            )));
        }
    }
    Ok(())
}

/// Remove `key` from `config` and return its value.
fn pop<'py>(config: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let value = config.get_item(key)?;
    if value.is_some() {
        config.del_item(key)?;
    }
    Ok(value)
}

/// Online score following of a MIDI performance (drop-in for `matchmaker.Matchmaker`).
#[pyclass(name = "Matchmaker", module = "matchmaker_rs", unsendable)]
struct PyMatchmaker {
    inner: CorePipeline,
    performance_path: PathBuf,
    // what `export` needs to build a second, recording pipeline
    method_kind: Method,
    method_config: MethodConfig,
    tempo_arg: Option<f64>,
    unfold: bool,
    started: bool,
    #[pyo3(get)]
    score_file: String,
    #[pyo3(get)]
    performance_file: String,
    #[pyo3(get)]
    input_type: String,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    tempo: f64,
    #[pyo3(get)]
    config: Py<PyDict>,
    #[pyo3(get)]
    polling_period: Option<f64>,
    #[pyo3(get, name = "_has_run")]
    has_run: bool,
}

enum CorePipeline {
    Midi(Box<Core>),
    Audio(Box<AudioMatchmaker>),
}

impl CorePipeline {
    fn score(&self) -> &Score {
        match self {
            Self::Midi(core) => core.score(),
            Self::Audio(core) => core.score(),
        }
    }

    fn tempo(&self) -> f64 {
        match self {
            Self::Midi(core) => core.tempo(),
            Self::Audio(core) => core.tempo(),
        }
    }

    fn alignment_path(&self) -> &AlignmentPath {
        match self {
            Self::Midi(core) => core.alignment_path(),
            Self::Audio(core) => core.alignment_path(),
        }
    }

    fn score_positions(&self) -> Vec<f32> {
        match self {
            Self::Midi(core) => core.score_positions(),
            Self::Audio(core) => core.score_positions().to_vec(),
        }
    }

    fn follower_score_positions(&self) -> &[f32] {
        match self {
            Self::Midi(core) => core.follower_score_positions(),
            Self::Audio(core) => core.score_positions(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn new_audio_matchmaker(
    py: Python<'_>,
    score_file: PathBuf,
    performance_file: Option<PathBuf>,
    method: Option<&str>,
    stream: Option<&Bound<'_, PyAny>>,
    processor: Option<&str>,
    device_name_or_index: Option<&Bound<'_, PyAny>>,
    tempo: Option<f64>,
    unfold_score: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyMatchmaker> {
    let method_name = method.unwrap_or("arzt");
    let audio_method: AudioMethod = method_name.parse().map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid method '{method_name}' for audio. Available: {}",
            available_audio_methods()
        ))
    })?;
    if audio_method != AudioMethod::Arzt {
        return Err(PyValueError::new_err(format!(
            "Audio method '{method_name}' is not implemented yet. Available: {}",
            available_audio_methods()
        )));
    }
    if stream.is_some() || device_name_or_index.is_some() {
        return Err(PyNotImplementedError::new_err(
            "audio devices and custom streams are not supported; pass `performance_file`",
        ));
    }
    let Some(performance_path) = performance_file else {
        return Err(PyNotImplementedError::new_err(
            "live audio input is not supported; pass `performance_file`",
        ));
    };

    let kwargs = match kwargs {
        Some(kwargs) => kwargs.copy()?,
        None => default_audio_kwargs(py, audio_method)?,
    };
    let mut audio_config = AudioMethodConfig::for_method(audio_method);
    let configured_processor: String = match pop(&kwargs, "processor")? {
        Some(value) => value.extract()?,
        None => audio_config.processor.name().into(),
    };
    let processor_name = processor.unwrap_or(&configured_processor);
    audio_config.processor = processor_name
        .parse::<AudioProcessorKind>()
        .map_err(to_py_err)?;

    let soundfont_path = match pop(&kwargs, "soundfont_path")? {
        Some(value) => Some(PathBuf::from(value.extract::<String>()?)),
        None => None,
    };
    for (key, value) in kwargs.iter() {
        let key = key.extract::<String>()?;
        set_audio_config(&mut audio_config, audio_method, &key, config_value(&value)?)?;
    }
    audio_config.soundfont_path = soundfont_path;
    audio_config.validate(audio_method).map_err(to_py_err)?;
    let inner = AudioMatchmaker::new_arzt(&score_file, audio_config, tempo, unfold_score)
        .map_err(to_py_err)?;

    Ok(PyMatchmaker {
        tempo: inner.tempo(),
        polling_period: None,
        inner: CorePipeline::Audio(Box::new(inner)),
        score_file: score_file.to_string_lossy().into_owned(),
        performance_file: performance_path.to_string_lossy().into_owned(),
        performance_path,
        method_kind: Method::Arzt,
        method_config: MethodConfig::for_method(Method::Arzt),
        tempo_arg: tempo,
        unfold: unfold_score,
        input_type: "audio".into(),
        method: method_name.into(),
        config: kwargs.unbind(),
        started: false,
        has_run: false,
    })
}

#[pymethods]
impl PyMatchmaker {
    #[new]
    #[pyo3(signature = (
        score_file,
        performance_file=None,
        input_type="audio",
        method=None,
        *,
        stream=None,
        processor=None,
        device_name_or_index=None,
        tempo=None,
        wait=false,
        unfold_score=true,
        kwargs=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        score_file: PathBuf,
        performance_file: Option<PathBuf>,
        input_type: &str,
        method: Option<&str>,
        stream: Option<&Bound<'_, PyAny>>,
        processor: Option<&str>,
        device_name_or_index: Option<&Bound<'_, PyAny>>,
        tempo: Option<f64>,
        wait: bool,
        unfold_score: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        // the simulation always runs as fast as possible
        let _ = wait;
        if input_type == "audio" {
            return new_audio_matchmaker(
                py,
                score_file,
                performance_file,
                method,
                stream,
                processor,
                device_name_or_index,
                tempo,
                unfold_score,
                kwargs,
            );
        }
        if input_type != "midi" {
            return Err(PyValueError::new_err(format!(
                "Invalid input_type '{input_type}'. Available input types: ['midi', 'audio']"
            )));
        }
        let method_name = method.unwrap_or("pthmm");
        let method_kind: Method = method_name.parse().map_err(|_| {
            PyValueError::new_err(format!(
                "Invalid method '{method_name}' for {input_type}. Available: {}",
                available_methods()
            ))
        })?;
        if stream.is_some() || device_name_or_index.is_some() {
            return Err(PyNotImplementedError::new_err(
                "custom streams and MIDI devices are not supported; pass `performance_file`",
            ));
        }

        // `kwargs` replaces the method's defaults, as in the reference
        let config = match kwargs {
            Some(kwargs) => kwargs.copy()?,
            None => default_kwargs(py, method_kind)?,
        };
        // replaced kwargs fall back to 10 ms windows; only the `event_based` method (outerhmm)
        // streams events
        let default_polling = (method_kind != Method::OuterHmm).then_some(0.01);
        let polling_period = match pop(&config, "polling_period")? {
            Some(value) if value.is_none() => None,
            Some(value) => Some(value.extract::<f64>()?),
            None => default_polling,
        };
        let configured_processor: String = match pop(&config, "processor")? {
            Some(value) => value.extract()?,
            None => "pitch".into(),
        };

        let mut method_config = MethodConfig::for_method(method_kind);
        method_config.polling_period = polling_period;
        let processor_name = processor.map_or(configured_processor, String::from);
        method_config
            .set(method_kind, "processor", &ConfigValue::Str(processor_name))
            .map_err(to_py_err)?;
        for (key, value) in config.iter() {
            let key = key.extract::<String>()?;
            // the reference's pass-through (`methods.yaml`) never hands these two to MIDI arzt
            if method_kind == Method::Arzt
                && matches!(key.as_str(), "window_size" | "start_window_size")
            {
                continue;
            }
            method_config
                .set(method_kind, &key, &config_value(&value)?)
                .map_err(to_py_err)?;
        }

        let inner = Core::with_config(
            &score_file,
            method_kind,
            method_config.clone(),
            tempo,
            unfold_score,
        )
        .map_err(|err| match err {
            Error::InvalidParameter { .. } => to_py_err(err),
            _ => PyValueError::new_err(format!("Invalid score file: {err}")),
        })?;

        let Some(performance_path) = performance_file else {
            return Err(PyNotImplementedError::new_err(
                "live MIDI input is not supported; pass `performance_file`",
            ));
        };
        let is_midi = performance_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "mid" | "midi"));
        if !is_midi {
            return Err(PyValueError::new_err(format!(
                "Invalid performance file. Expected MIDI file, but got {}",
                performance_path.display()
            )));
        }

        Ok(PyMatchmaker {
            tempo: inner.tempo(),
            polling_period: inner.polling_period(),
            inner: CorePipeline::Midi(Box::new(inner)),
            score_file: score_file.to_string_lossy().into_owned(),
            performance_file: performance_path.to_string_lossy().into_owned(),
            performance_path,
            method_kind,
            method_config,
            tempo_arg: tempo,
            unfold: unfold_score,
            input_type: input_type.to_string(),
            method: method_name.to_string(),
            config: config.unbind(),
            started: false,
            has_run: false,
        })
    }

    /// Follow the performance; yields the score position in beats per observation and returns
    /// the alignment path `(2, T)` (the `StopIteration` value).
    #[pyo3(signature = (verbose=true))]
    fn run(slf: &Bound<'_, Self>, verbose: bool) -> PyResult<Run> {
        let _ = verbose;
        let mut this = slf.borrow_mut();
        if this.started {
            return Err(PyRuntimeError::new_err(
                "Matchmaker.run() can only be called once",
            ));
        }
        this.started = true;
        Ok(Run {
            matchmaker: slf.clone().unbind(),
            messages: None,
            cursor: 0,
            pending: VecDeque::new(),
            audio_positions: None,
            finished: false,
        })
    }

    /// The follower: `alignment_path`, `_alignment_path` and `score_positions` like the reference.
    #[getter]
    fn score_follower(slf: &Bound<'_, Self>) -> ScoreFollower {
        ScoreFollower {
            matchmaker: slf.clone().unbind(),
        }
    }

    /// Ascending score beat of every note onset.
    #[getter]
    fn score_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.score_positions().into_pyarray(py)
    }

    /// Re-run the performance with recording and write the intermediate data in the layout of
    /// the golden data (compare with `tools/golden/compare.py`).
    fn export(&self, directory: PathBuf) -> PyResult<()> {
        let CorePipeline::Midi(_) = self.inner else {
            return Err(PyNotImplementedError::new_err(
                "audio export is not supported yet",
            ));
        };
        let mut recorded = Core::with_config(
            &self.score_file,
            self.method_kind,
            self.method_config.clone(),
            self.tempo_arg,
            self.unfold,
        )
        .map_err(to_py_err)?
        .record();
        recorded
            .run_file(&self.performance_path)
            .map_err(to_py_err)?;
        write_run(&recorded, directory).map_err(to_py_err)
    }

    /// Score annotations: whole beats (`level="beat"`) or every onset (`"note"`), in beats
    /// (float32) or seconds at the tempo (float64).
    #[pyo3(signature = (level="beat", musical_beat=false, return_type="beats"))]
    fn build_score_annotations<'py>(
        &self,
        py: Python<'py>,
        level: &str,
        musical_beat: bool,
        return_type: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let level_kind = match level {
            "beat" => AnnotationLevel::Beat,
            "note" => AnnotationLevel::Note,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid score annotation level: {level}"
                )));
            }
        };
        let score = self.inner.score();
        match return_type {
            // `np.arange` gives float64 for whole beats; note onsets stay float32
            "beats" if level_kind == AnnotationLevel::Beat => Ok(score
                .annotations(level_kind, musical_beat)
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
                .into_pyarray(py)
                .into_any()),
            "beats" => Ok(score
                .annotations(level_kind, musical_beat)
                .into_pyarray(py)
                .into_any()),
            "seconds" => Ok(score
                .annotations_seconds(level_kind, musical_beat, self.inner.tempo())
                .into_pyarray(py)
                .into_any()),
            _ => Err(PyValueError::new_err(format!(
                "Invalid return type: {return_type}"
            ))),
        }
    }
}

/// Iterator returned by `Matchmaker.run`: feeds the performance message by message.
#[pyclass(module = "matchmaker_rs")]
struct Run {
    matchmaker: Py<PyMatchmaker>,
    messages: Option<Vec<MidiMessage>>,
    cursor: usize,
    pending: VecDeque<f64>,
    audio_positions: Option<Vec<f64>>,
    finished: bool,
}

#[pymethods]
impl Run {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<f64> {
        loop {
            if let Some(position) = self.pending.pop_front() {
                return Ok(position);
            }
            let mut matchmaker = self.matchmaker.bind(py).borrow_mut();
            if self.finished {
                let path = path_array(py, matchmaker.inner.alignment_path());
                return Err(PyStopIteration::new_err(path.unbind()));
            }
            if let Some(positions) = &self.audio_positions {
                if let Some(&position) = positions.get(self.cursor) {
                    self.cursor += 1;
                    return Ok(position);
                }
                self.finished = true;
                matchmaker.has_run = true;
                let path = path_array(py, matchmaker.inner.alignment_path());
                return Err(PyStopIteration::new_err(path.unbind()));
            }
            let performance_path = matchmaker.performance_path.clone();
            match &mut matchmaker.inner {
                CorePipeline::Audio(audio) => {
                    self.audio_positions =
                        Some(audio.run_file(&performance_path).map_err(to_py_err)?);
                }
                CorePipeline::Midi(core) => {
                    let messages = match &self.messages {
                        Some(messages) => messages,
                        None => {
                            let loaded = Performance::load(&performance_path)
                                .map_err(to_py_err)?
                                .messages();
                            self.messages.insert(loaded)
                        }
                    };
                    match messages.get(self.cursor) {
                        Some(&message) => {
                            self.cursor += 1;
                            self.pending.extend(core.push(message).map_err(to_py_err)?);
                        }
                        None => {
                            self.pending.extend(core.finish());
                            self.finished = true;
                            matchmaker.has_run = true;
                        }
                    }
                }
            }
        }
    }
}

/// The reference's `score_follower` attributes, read from the running `Matchmaker`.
#[pyclass(module = "matchmaker_rs", unsendable)]
struct ScoreFollower {
    matchmaker: Py<PyMatchmaker>,
}

#[pymethods]
impl ScoreFollower {
    /// `(2, T)` float64: performance seconds and score beats.
    #[getter]
    fn alignment_path<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        path_array(py, self.matchmaker.bind(py).borrow().inner.alignment_path())
    }

    /// The path as a list of `(perf_time, score_beat)` tuples.
    #[getter(_alignment_path)]
    fn alignment_path_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let matchmaker = self.matchmaker.bind(py).borrow();
        let path = matchmaker.inner.alignment_path();
        PyList::new(
            py,
            path.perf_time
                .iter()
                .copied()
                .zip(path.score_beat.iter().copied()),
        )
    }

    /// The follower's states in beats (float32, with inserted states for `pthmm` and `hmm`).
    #[getter]
    fn score_positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let matchmaker = self.matchmaker.bind(py).borrow();
        matchmaker
            .inner
            .follower_score_positions()
            .to_vec()
            .into_pyarray(py)
    }
}

#[pymodule]
#[pyo3(name = "matchmaker_rs")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyMatchmaker>()?;

    let methods = PyDict::new(py);
    methods.set_item("midi", Method::AVAILABLE.to_vec())?;
    methods.set_item("audio", vec!["arzt"])?;
    m.add("AVAILABLE_METHODS", methods)?;

    let defaults = PyDict::new(py);
    let midi = PyDict::new(py);
    for name in Method::AVAILABLE {
        midi.set_item(name, default_kwargs(py, name.parse().map_err(to_py_err)?)?)?;
    }
    defaults.set_item("midi", midi)?;
    let audio = PyDict::new(py);
    audio.set_item("arzt", default_audio_kwargs(py, AudioMethod::Arzt)?)?;
    defaults.set_item("audio", audio)?;
    m.add("DEFAULT_KWARGS", defaults)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__reference_version__", REFERENCE_VERSION)?;
    Ok(())
}
