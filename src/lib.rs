//! Pure-Rust port of pymatchmaker (real-time music alignment / score following).

/// Placeholder until the first feature lands.
///
/// ```
/// assert_eq!(matchmaker_rs::add(2, 2), 4);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
