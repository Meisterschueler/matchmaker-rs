//! Port of `matchmaker.dp`: the event-level online time warping followers `arzt` and `dixon`
//! and what they share.
//!
//! Upstream: `matchmaker/dp/direction.py` (`Direction`); the metrics are the SciPy `cdist` ones the
//! followers call. `dtw_loop.pyx` is not ported: the event classes have their own loop (it
//! belongs to the frame-level audio variant).
//!
//! ```
//! use matchmaker_rs::dp::Distance;
//! let d = Distance::Cosine.between(&[1.0, 0.0], &[1.0, 1.0]);
//! assert!((d - 0.292_893_218_813_452_5).abs() < 1e-12);
//! ```

pub mod arzt;
pub mod arzt_frame;
pub mod dixon;

use std::str::FromStr;

use crate::error::Error;

/// Port of `matchmaker.dp.direction.Direction` (`NONE` is unused here): the axes a step advances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Advance the reference (score) only.
    Ref,
    /// Advance the input (performance) only.
    Input,
    /// Advance both.
    Both,
}

/// The `scipy.spatial.distance.cdist` metrics the followers support (`distance_func`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// `1 - u·v / (‖u‖‖v‖)`, NaN if either vector is all zero.
    Cosine,
    /// `‖u - v‖`.
    Euclidean,
}

impl Distance {
    /// Names accepted by [`FromStr`].
    pub const AVAILABLE: [&'static str; 2] = ["cosine", "euclidean"];

    /// `cdist(u, v, metric)` for one pair; `float32` inputs are widened to `float64` first.
    ///
    /// The sums run in order; for the binary vectors of the MIDI pipeline every sum is a small
    /// integer, so the result equals SciPy's bit for bit (002 research R4).
    pub fn between(self, u: &[f32], v: &[f32]) -> f64 {
        let pairs = || u.iter().zip(v).map(|(&a, &b)| (f64::from(a), f64::from(b)));
        match self {
            Distance::Cosine => {
                let (mut dot, mut uu, mut vv) = (0.0, 0.0, 0.0);
                for (a, b) in pairs() {
                    dot += a * b;
                    uu += a * a;
                    vv += b * b;
                }
                // SciPy: cosine /= nu * nv, clipped to [-1, 1]
                let cosine = dot / (uu.sqrt() * vv.sqrt());
                1.0 - if cosine.abs() > 1.0 {
                    1.0f64.copysign(cosine)
                } else {
                    cosine
                }
            }
            Distance::Euclidean => pairs().map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt(),
        }
    }
}

impl FromStr for Distance {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        match name {
            "cosine" => Ok(Distance::Cosine),
            "euclidean" => Ok(Distance::Euclidean),
            _ => Err(Error::InvalidParameter {
                key: "distance_func".into(),
                reason: format!(
                    "unknown distance {name:?}; available: {:?}",
                    Self::AVAILABLE
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_matches_scipy_examples() {
        // scipy.spatial.distance.cdist on float32 rows, see 002 research R4
        let a = [1.0, 1.0, 1.0, 0.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            Distance::Cosine.between(&[1.0, 1.0, 0.0], &[0.0, 1.0, 0.0]),
            0.292_893_218_813_452_54
        );
        assert_eq!(Distance::Cosine.between(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
        // 3 / (sqrt(3) * sqrt(3)) rounds above 1 and is clipped: the distance is exactly 0
        assert_eq!(Distance::Cosine.between(&a, &a), 0.0);
        assert!((Distance::Cosine.between(&a, &b) - (1.0 - 1.0 / 6f64.sqrt())).abs() < 1e-15);
    }

    #[test]
    fn zero_vector_cosine_is_nan_but_euclidean_is_defined() {
        let (zero, a) = ([0.0; 4], [1.0, 0.0, 1.0, 1.0]);
        assert!(Distance::Cosine.between(&zero, &a).is_nan());
        assert!(Distance::Cosine.between(&a, &zero).is_nan());
        assert!(Distance::Cosine.between(&zero, &zero).is_nan());
        assert_eq!(Distance::Euclidean.between(&zero, &a), 3f64.sqrt());
        assert_eq!(Distance::Euclidean.between(&zero, &zero), 0.0);
    }

    #[test]
    fn names_parse_and_unknown_names_list_the_choices() {
        assert_eq!("cosine".parse::<Distance>().unwrap(), Distance::Cosine);
        assert_eq!(
            "euclidean".parse::<Distance>().unwrap(),
            Distance::Euclidean
        );
        let error = "manhattan".parse::<Distance>().unwrap_err().to_string();
        assert!(
            error.contains("distance_func") && error.contains("cosine"),
            "{error}"
        );
    }
}
