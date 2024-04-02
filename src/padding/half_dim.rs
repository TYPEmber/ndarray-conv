use ndarray::{
    Array, ArrayBase, Axis, Data, Dim, Dimension, IntoDimension, Ix, OwnedRepr, RemoveAxis,
    SliceArg, SliceInfo, SliceInfoElem,
};

#[inline]
pub fn constant_front<const N: usize, T>(
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
    for j in 0..padding[0] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).fill(constant);
        }
    }
}

#[inline]
pub fn constant_back<const N: usize, T>(
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
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).fill(constant);
        }
    }
}

#[inline]
pub fn replicate_front<const N: usize, T>(
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
    let border = buffer.index_axis(Axis(dim), padding[0]);
    for j in 0..padding[0] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).assign(&border);
        }
    }
}

#[inline]
pub fn replicate_back<const N: usize, T>(
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
    let border = buffer.index_axis(Axis(dim), buffer.raw_dim()[dim] - padding[1] - 1);
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        unsafe {
            let buffer_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            buffer_mut.index_axis_mut(Axis(dim), j).assign(&border);
        }
    }
}

#[inline]
pub fn reflect_front<const N: usize, T>(
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
    let border_index = padding[0];
    for j in 0..padding[0] {
        let reflect_j = (border_index - j) + border_index;
        unsafe {
            let output_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), reflect_j));
        }
    }
}

#[inline]
pub fn reflect_back<const N: usize, T>(
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
    let border_index = buffer.raw_dim()[dim] - padding[1] - 1;
    for j in input_dim[dim] + padding[0]..buffer.raw_dim()[dim] {
        let reflect_j = border_index - (j - border_index);
        unsafe {
            let output_mut = (buffer as *const _ as *mut Array<T, Dim<[Ix; N]>>)
                .as_mut()
                .unwrap();

            output_mut
                .index_axis_mut(Axis(dim), j)
                .assign(&buffer.index_axis(Axis(dim), reflect_j));
        }
    }
}

#[inline]
pub fn circular_front<const N: usize, T>(
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
    let border_index = padding[0];
    for j in 0..padding[0] {
        let circular_j = buffer.raw_dim()[dim] - padding[1] - (border_index - j);
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

#[inline]
pub fn circular_back<const N: usize, T>(
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
