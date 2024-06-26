use ndarray::{ArrayBase, DataMut, Dim, Ix, RemoveAxis};
use num::traits::NumAssign;

use super::half_dim;

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
