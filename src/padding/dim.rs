//! Provides padding functions for individual dimensions.
//!
//! This module contains functions for applying padding to a specific
//! dimension of an array, including constant, replicate, reflect, and
//! circular padding modes. These functions are used internally by the
//! `PaddingExt` trait to implement padding in N-dimensional arrays.

use ndarray::{ArrayBase, DataMut, Dim, Ix, RemoveAxis};
use num::traits::NumAssign;

use super::half_dim;

/// Applies constant padding to a specific dimension of the input array.
///
/// This function pads the front and back of a given dimension with a
/// specified constant value.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the input data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
/// * `constant`: The constant value to pad with.
#[inline]
pub fn constant<const N: usize, T, S, D, DO>(
    input_dim: D,
    buffer: &mut ArrayBase<S, DO>,
    dim: usize,
    padding: [usize; 2],
    constant: T,
) where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
    DO: RemoveAxis,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::constant_front(buffer, dim, padding, constant);
    half_dim::constant_back(input_dim, buffer, dim, padding, constant);
}

/// Applies replicate padding to a specific dimension of the input array.
///
/// This function pads the front and back of a given dimension by replicating
/// the edge values.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the input data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn replicate<const N: usize, T, S, D, DO>(
    input_dim: D,
    buffer: &mut ArrayBase<S, DO>,
    dim: usize,
    padding: [usize; 2],
) where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
    DO: RemoveAxis,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::replicate_front(buffer, dim, padding);
    half_dim::replicate_back(input_dim, buffer, dim, padding);
}

/// Applies reflect padding to a specific dimension of the input array.
///
/// This function pads the front and back of a given dimension by
/// reflecting the array at the boundaries.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the input data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn reflect<const N: usize, T, S, D, DO>(
    input_dim: D,
    buffer: &mut ArrayBase<S, DO>,
    dim: usize,
    padding: [usize; 2],
) where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
    DO: RemoveAxis,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::reflect_front(buffer, dim, padding);
    half_dim::reflect_back(input_dim, buffer, dim, padding);
}

/// Applies circular padding to a specific dimension of the input array.
///
/// This function pads the front and back of a given dimension by
/// wrapping the data around the boundaries.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the input data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn circular<const N: usize, T, S, D, DO>(
    input_dim: D,
    buffer: &mut ArrayBase<S, DO>,
    dim: usize,
    padding: [usize; 2],
) where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
    DO: RemoveAxis,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::circular_front(buffer, dim, padding);
    half_dim::circular_back(input_dim, buffer, dim, padding);
}
