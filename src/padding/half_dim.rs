//! Provides functions for padding on a single side of a dimension.
//!
//! This module contains functions for applying padding to the front
//! or back of a specific dimension of an array. These functions are
//! used internally by the `dim` module and implement the core logic
//! for different padding modes like constant, replicate, reflect and
//! circular.

use ndarray::{Array, ArrayBase, Axis, DataMut, Dim, Ix, RemoveAxis};
use num::traits::NumAssign;

/// Applies constant padding to the front of a given dimension of an array.
///
/// This function modifies the input array by padding the front of the
/// specified dimension with a constant value.
///
/// # Type Parameters
///
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the array.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
/// * `constant`: The constant value to pad with.
#[inline]
pub fn constant_front<T, S, D>(
    buffer: &mut ArrayBase<S, D>,
    dim: usize,
    padding: [usize; 2],
    constant: T,
) where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
{
    for j in 0..padding[0] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).fill(constant);
        }
    }
}

/// Applies constant padding to the back of a given dimension of an array.
///
/// This function modifies the input array by padding the back of the
/// specified dimension with a constant value.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the original data.
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
pub fn constant_back<const N: usize, T, S, D, DO>(
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
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut ArrayBase<S, DO>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).fill(constant);
        }
    }
}

/// Applies replicate padding to the front of a given dimension of an array.
///
/// This function modifies the input array by padding the front of the
/// specified dimension by replicating the edge values.
///
/// # Type Parameters
///
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the array.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn replicate_front<T, S, D>(buffer: &mut ArrayBase<S, D>, dim: usize, padding: [usize; 2])
where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
{
    let border = buffer.index_axis(Axis(dim), padding[0]);
    for j in 0..padding[0] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).assign(&border);
        }
    }
}

/// Applies replicate padding to the back of a given dimension of an array.
///
/// This function modifies the input array by padding the back of the
/// specified dimension by replicating the edge values.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the original data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn replicate_back<const N: usize, T, S, D, DO>(
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
    let border = buffer.index_axis(Axis(dim), buffer.raw_dim()[dim] - padding[1] - 1);
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).assign(&border);
        }
    }
}

/// Applies reflect padding to the front of a given dimension of an array.
///
/// This function modifies the input array by padding the front of the
/// specified dimension by reflecting the array at the boundaries.
///
/// # Type Parameters
///
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the array.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn reflect_front<T, S, D>(buffer: &mut ArrayBase<S, D>, dim: usize, padding: [usize; 2])
where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
{
    let border_index = padding[0];
    for j in 0..padding[0] {
        let reflect_j = (border_index - j) + border_index;
        unsafe {
            let output_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), reflect_j));
        }
    }
}

/// Applies reflect padding to the back of a given dimension of an array.
///
/// This function modifies the input array by padding the back of the
/// specified dimension by reflecting the array at the boundaries.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the original data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn reflect_back<const N: usize, T, S, D, DO>(
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
    let border_index = buffer.raw_dim()[dim] - padding[1] - 1;
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        let reflect_j = border_index - (j - border_index);
        unsafe {
            let output_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), reflect_j));
        }
    }
}

/// Applies circular padding to the front of a given dimension of an array.
///
/// This function modifies the input array by padding the front of the
/// specified dimension by wrapping the data around the boundaries.
///
/// # Type Parameters
///
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the array.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn circular_front<T, S, D>(buffer: &mut ArrayBase<S, D>, dim: usize, padding: [usize; 2])
where
    T: NumAssign + Copy,
    S: DataMut<Elem = T>,
    D: RemoveAxis,
{
    let border_index = padding[0];
    for j in 0..padding[0] {
        let circular_j = buffer.raw_dim()[dim] - padding[1] - (border_index - j);
        unsafe {
            let output_mut = (buffer as *const _ as *mut ArrayBase<S, D>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), circular_j));
        }
    }
}

/// Applies circular padding to the back of a given dimension of an array.
///
/// This function modifies the input array by padding the back of the
/// specified dimension by wrapping the data around the boundaries.
///
/// # Type Parameters
///
/// * `N`: The number of dimensions.
/// * `T`: The numeric type of the array elements.
/// * `S`: The data storage type of the array.
/// * `D`: The dimension type of the original data.
/// * `DO`: The dimension type of the output data.
///
/// # Arguments
///
/// * `input_dim`: The dimensions of the original array.
/// * `buffer`: A mutable reference to the array to be padded.
/// * `dim`: The dimension to pad.
/// * `padding`: An array containing two `usize` values representing padding for front and back, respectively.
#[inline]
pub fn circular_back<const N: usize, T, S, D, DO>(
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
    let border_index = buffer.raw_dim()[dim] - padding[1] - 1;
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        let circular_j = padding[0] + (j - border_index - 1);
        unsafe {
            let output_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), circular_j));
        }
    }
}
