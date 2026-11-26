//! NumPy's scalar indirect quicksort, applied to MIDI messages by timestamp.
//!
//! Derived from NumPy's `numpy/_core/src/npysort/quicksort.hpp` and
//! `npysort_heapsort.h`. Copyright (c) 2005-2025 NumPy Developers;
//! redistribution is governed by `NUMPY-LICENSE`.

use crate::io::stream::MidiMessage;

fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

fn sift_down(messages: &mut [MidiMessage], start: usize, end: usize) {
    let value = messages[start];
    let mut root = start;
    while root < end / 2 {
        let mut child = 2 * root + 1;
        if child + 1 < end && less(messages[child].time, messages[child + 1].time) {
            child += 1;
        }
        if !less(value.time, messages[child].time) {
            break;
        }
        messages[root] = messages[child];
        root = child;
    }
    messages[root] = value;
}

fn heapsort(messages: &mut [MidiMessage]) {
    for root in (0..messages.len() / 2).rev() {
        sift_down(messages, root, messages.len());
    }
    for end in (1..messages.len()).rev() {
        messages.swap(0, end);
        sift_down(messages, 0, end);
    }
}

pub(crate) fn sort_messages(messages: &mut [MidiMessage]) {
    if messages.len() < 2 {
        return;
    }
    let mut left = 0;
    let mut right = messages.len() - 1;
    let mut depth = 2 * (usize::BITS - 1 - messages.len().leading_zeros()) as i32;
    let mut stack = Vec::new();
    loop {
        if depth < 0 {
            heapsort(&mut messages[left..=right]);
        } else {
            while right - left > 15 {
                let middle = left + (right - left) / 2;
                if less(messages[middle].time, messages[left].time) {
                    messages.swap(middle, left);
                }
                if less(messages[right].time, messages[middle].time) {
                    messages.swap(right, middle);
                }
                if less(messages[middle].time, messages[left].time) {
                    messages.swap(middle, left);
                }
                let pivot = messages[middle].time;
                let mut i = left;
                let mut j = right - 1;
                messages.swap(middle, j);
                loop {
                    i += 1;
                    while less(messages[i].time, pivot) {
                        i += 1;
                    }
                    j -= 1;
                    while less(pivot, messages[j].time) {
                        j -= 1;
                    }
                    if i >= j {
                        break;
                    }
                    messages.swap(i, j);
                }
                messages.swap(i, right - 1);
                depth -= 1;
                if i - left < right - i {
                    stack.push((i + 1, right, depth));
                    right = i - 1;
                } else {
                    stack.push((left, i - 1, depth));
                    left = i + 1;
                }
            }
            for i in left + 1..=right {
                let value = messages[i];
                let mut j = i;
                while j > left && less(value.time, messages[j - 1].time) {
                    messages[j] = messages[j - 1];
                    j -= 1;
                }
                messages[j] = value;
            }
        }
        let Some((next_left, next_right, next_depth)) = stack.pop() else {
            break;
        };
        (left, right, depth) = (next_left, next_right, next_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::stream::MidiMessageKind;

    fn messages(n: u8, modulus: u8) -> Vec<MidiMessage> {
        (0..n)
            .map(|index| MidiMessage {
                kind: MidiMessageKind::NoteOn {
                    note: index,
                    velocity: 1,
                },
                channel: 0,
                time: f64::from((u16::from(index) * 17 % u16::from(modulus)) as u8),
            })
            .collect()
    }

    fn pitches(messages: &[MidiMessage]) -> Vec<u8> {
        messages
            .iter()
            .map(|message| message.note_on_pitch().unwrap())
            .collect()
    }

    #[test]
    fn scalar_quicksort_matches_numpy_with_equal_times() {
        for (n, expected) in [
            (
                16,
                vec![0, 11, 2, 13, 4, 15, 6, 8, 10, 1, 12, 3, 14, 5, 7, 9],
            ),
            (
                17,
                vec![0, 11, 13, 2, 4, 15, 6, 8, 10, 12, 1, 3, 14, 5, 16, 7, 9],
            ),
            (
                48,
                vec![
                    0, 44, 22, 33, 11, 24, 46, 35, 13, 2, 4, 37, 26, 15, 39, 17, 28, 6, 8, 19, 41,
                    30, 32, 21, 10, 43, 45, 34, 23, 1, 12, 47, 25, 3, 14, 36, 5, 27, 38, 16, 40, 7,
                    18, 29, 9, 42, 20, 31,
                ],
            ),
        ] {
            let mut input = messages(n, 11);
            sort_messages(&mut input);
            assert_eq!(pitches(&input), expected);
        }
        sort_messages(&mut []);
    }

    #[test]
    fn quicksort_fallback_matches_numpy_heapsort_with_equal_times() {
        let mut input = messages(32, 7);
        heapsort(&mut input);
        assert_eq!(
            pitches(&input),
            [
                21, 0, 14, 7, 28, 5, 19, 26, 12, 10, 24, 17, 3, 31, 22, 29, 8, 1, 15, 6, 27, 13,
                20, 25, 11, 4, 18, 30, 23, 2, 9, 16
            ]
        );
    }

    #[test]
    fn scalar_quicksort_handles_signed_zero_and_nan_like_numpy() {
        let mut input = messages(8, 11);
        for (message, time) in input.iter_mut().zip([
            f64::NAN,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            f64::NAN,
            0.0,
        ]) {
            message.time = time;
        }
        sort_messages(&mut input);
        assert_eq!(pitches(&input), [4, 1, 2, 7, 5, 3, 0, 6]);
    }
}
