use ndarray::{
    ArrayBase, Dim, Dimension, IntoDimension, Ix, OwnedRepr, RemoveAxis, SliceArg, SliceInfo,
    SliceInfoElem,
};

use super::half_dim;

#[inline]
pub fn constant<const N: usize, T>(
    input_dim: Dim<[usize; N]>,
    buffer: &mut ArrayBase<OwnedRepr<T>, Dim<[usize; N]>>,
    dim: usize,
    padding: [usize; 2],
    constant: T,
) where
    T: num::traits::NumAssign + Copy + Clone,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::constant_front(buffer, dim, padding, constant);
    half_dim::constant_back(input_dim, buffer, dim, padding, constant);
}

#[inline]
pub fn replicate<const N: usize, T>(
    input_dim: Dim<[usize; N]>,
    buffer: &mut ArrayBase<OwnedRepr<T>, Dim<[usize; N]>>,
    dim: usize,
    padding: [usize; 2],
) where
    T: num::traits::NumAssign + Copy + Clone,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::replicate_front(buffer, dim, padding);
    half_dim::replicate_back(input_dim, buffer, dim, padding);
}

#[inline]
pub fn reflect<const N: usize, T>(
    input_dim: Dim<[usize; N]>,
    buffer: &mut ArrayBase<OwnedRepr<T>, Dim<[usize; N]>>,
    dim: usize,
    padding: [usize; 2],
) where
    T: num::traits::NumAssign + Copy + Clone,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::reflect_front(buffer, dim, padding);
    half_dim::reflect_back(input_dim, buffer, dim, padding);
}

#[inline]
pub fn circular<const N: usize, T>(
    input_dim: Dim<[usize; N]>,
    buffer: &mut ArrayBase<OwnedRepr<T>, Dim<[usize; N]>>,
    dim: usize,
    padding: [usize; 2],
) where
    T: num::traits::NumAssign + Copy + Clone,
    Dim<[Ix; N]>: Dimension,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>: SliceArg<Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
{
    half_dim::circular_front(buffer, dim, padding);
    half_dim::circular_back(input_dim, buffer, dim, padding);
}
