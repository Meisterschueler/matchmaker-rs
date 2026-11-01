//! `numpy.sum` for contiguous `float64` data: the same pairwise summation, so sums round
//! exactly like the reference's.
//!
//! Upstream: NumPy `numpy/_core/src/umath/loops_utils.h.src` (`pairwise_sum`); NumPy sums
//! with it along the contiguous axis, so `x.sum(1)` of a row-major matrix is `np_sum` per row.
//!
//! ```
//! use matchmaker_rs::utils::np_sum::np_sum;
//! let x: Vec<f64> = (1..=1000).map(|i| 1.0 / f64::from(i)).collect();
//! assert_eq!(np_sum(&x), 7.485470860550345); // `np.sum`; a left-to-right sum gives ...343
//! ```

/// `np.sum(a)` of a contiguous `float64` array.
pub fn np_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |sum, &x| sum + x);
    }
    if n <= BLOCK {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let blocks = n - n % 8;
        for i in (8..blocks).step_by(8) {
            for (acc, &x) in r.iter_mut().zip(&a[i..i + 8]) {
                *acc += x;
            }
        }
        let paired = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return a[blocks..].iter().fold(paired, |sum, &x| sum + x);
    }
    let half = n / 2;
    let half = half - half % 8;
    np_sum(&a[..half]) + np_sum(&a[half..])
}

#[cfg(test)]
mod tests {
    use super::np_sum;

    /// `np.sum(1 / (arange(n) + 1))` from NumPy 2.5.4 (n = 128 and 129 differ from a
    /// left-to-right sum in the last digit).
    #[test]
    fn matches_numpy() {
        for (n, expected) in [
            (0, 0.0),
            (5, 2.283333333333333),
            (8, 2.7178571428571425),
            (100, 5.187377517639621),
            (128, 5.433147092589173),
            (129, 5.440899030573669),
            (1000, 7.485470860550345),
            (3000, 8.583749889959186),
        ] {
            let x: Vec<f64> = (1..=n).map(|i| 1.0 / f64::from(i)).collect();
            assert_eq!(np_sum(&x), expected, "n = {n}");
        }
    }
}
