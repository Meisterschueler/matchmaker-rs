//! Python bindings of matchmaker-rs: a drop-in `Matchmaker` for pymatchmaker's MIDI usage.
//!
//! Ports the public surface of `matchmaker.matchmaker.Matchmaker` (MIDI methods only).
//! See `specs/001-midi-score-following/contracts/python-api.md`.

use std::collections::VecDeque;
use std::path::PathBuf;

use matchmaker_rs::Error;
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::io::stream::MidiMessage;
use matchmaker_rs::matchmaker::{
    AnnotationLevel, ConfigValue, Matchmaker as Core, Method, MethodConfig,
};
use matchmaker_rs::performance::Performance;
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

/// Errors about parameters are `ValueError`; a file that cannot be read is `OSError`.
fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Io { .. } => PyOSError::new_err(err.to_string()),
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

/// `DEFAULT_KWARGS["midi"][method]`.
fn default_kwargs<'py>(py: Python<'py>, method: Method) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("processor", "pitch")?;
    kwargs.set_item("piano_range", true)?;
    if method == Method::OuterHmm {
        kwargs.set_item("polling_period", 0.001)?;
    }
    Ok(kwargs)
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
    inner: Core,
    performance_path: PathBuf,
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
        if input_type != "midi" {
            return Err(PyValueError::new_err(format!(
                "Invalid input_type '{input_type}': matchmaker-rs supports MIDI input only. \
                 Available methods for midi: {}",
                available_methods()
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
            method_config
                .set(
                    method_kind,
                    &key.extract::<String>()?,
                    &config_value(&value)?,
                )
                .map_err(to_py_err)?;
        }

        let inner = Core::with_config(&score_file, method_kind, method_config, tempo, unfold_score)
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
            inner,
            score_file: score_file.to_string_lossy().into_owned(),
            performance_file: performance_path.to_string_lossy().into_owned(),
            performance_path,
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

    /// Score annotations in beats: whole beats (`level="beat"`) or every onset (`"note"`).
    #[pyo3(signature = (level="beat", musical_beat=false, return_type="beats"))]
    fn build_score_annotations<'py>(
        &self,
        py: Python<'py>,
        level: &str,
        musical_beat: bool,
        return_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let level_kind = match level {
            "beat" => AnnotationLevel::Beat,
            "note" => AnnotationLevel::Note,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid score annotation level: {level}"
                )));
            }
        };
        if musical_beat && level_kind == AnnotationLevel::Beat {
            return Err(PyNotImplementedError::new_err(
                "musical_beat=True is not supported",
            ));
        }
        match return_type {
            "beats" => Ok(self.inner.score_annotations(level_kind).into_pyarray(py)),
            "seconds" => Err(PyNotImplementedError::new_err(
                "return_type='seconds' is not supported; use 'beats'",
            )),
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
            let messages = match &self.messages {
                Some(messages) => messages,
                None => {
                    let loaded = Performance::load(&matchmaker.performance_path)
                        .map_err(to_py_err)?
                        .messages();
                    self.messages.insert(loaded)
                }
            };
            match messages.get(self.cursor) {
                Some(&message) => {
                    self.cursor += 1;
                    let positions = matchmaker.inner.push(message).map_err(to_py_err)?;
                    self.pending.extend(positions);
                }
                None => {
                    self.pending.extend(matchmaker.inner.finish());
                    self.finished = true;
                    matchmaker.has_run = true;
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
    m.add_class::<ScoreFollower>()?;
    m.add_class::<Run>()?;

    let methods = PyDict::new(py);
    methods.set_item("midi", Method::AVAILABLE.to_vec())?;
    m.add("AVAILABLE_METHODS", methods)?;

    let defaults = PyDict::new(py);
    let midi = PyDict::new(py);
    for name in Method::AVAILABLE {
        midi.set_item(name, default_kwargs(py, name.parse().map_err(to_py_err)?)?)?;
    }
    defaults.set_item("midi", midi)?;
    m.add("DEFAULT_KWARGS", defaults)?;

    let default_method = PyDict::new(py);
    default_method.set_item("midi", "pthmm")?;
    m.add("DEFAULT_METHOD", default_method)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__reference_version__", REFERENCE_VERSION)?;
    Ok(())
}
