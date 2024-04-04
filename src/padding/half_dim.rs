use ndarray::{Array, ArrayBase, Axis, DataMut, Dim, Ix, RemoveAxis};
use num::traits::NumAssign;

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
