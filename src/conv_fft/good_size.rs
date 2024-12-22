//! Provides functions for determining good FFT sizes.
//!
//! This module implements strategies for finding FFT sizes that
//! are efficient for the `rustfft` library.

fn good_size_cc(n: usize) -> usize {
    let mut best_fac = n.next_power_of_two();

    loop {
        let new_fac = best_fac / 4 * 3;
        match new_fac.cmp(&n) {
            std::cmp::Ordering::Less => break,
            std::cmp::Ordering::Equal => return n,
            std::cmp::Ordering::Greater => {
                best_fac = new_fac;
            }
        }
    }
    loop {
        let new_fac = best_fac / 6 * 5;
        match new_fac.cmp(&n) {
            std::cmp::Ordering::Less => break,
            std::cmp::Ordering::Equal => return n,
            std::cmp::Ordering::Greater => {
                best_fac = new_fac;
            }
        }
    }

    best_fac
}

/// Computes efficient FFT sizes for each dimension of an array.
///
/// This function takes an array of dimensions and returns an array of the same
/// size where each element is a "good" size for FFT calculations, as determined
/// by the `good_size_cc` function.
///
/// This function seems to be very slow for numbers that have large prime components.
pub fn compute<const N: usize>(size: &[usize; N]) -> [usize; N] {
    std::array::from_fn(|i| good_size_cc(size[i]))
}
