use ndarray::{ArrayBase, Data, Dim, Dimension, IntoDimension, Ix, RawData};

pub struct KernelWithDilation<S: RawData, const N: usize> {
    pub kernel: ArrayBase<S, Dim<[Ix; N]>>,
    pub dilation: [usize; N],
}

impl<S: RawData, const N: usize, T> KernelWithDilation<S, N>
where
    T: num::traits::NumAssign + Copy,
    S: Data<Elem = T>,
    Dim<[Ix; N]>: Dimension,
{
    pub fn gen_offset_list(&self, shape: &[usize]) -> Vec<(isize, T)> {
        let mut strides: [isize; N] = [0; N];

        strides
            .iter_mut()
            .take(N)
            .zip(shape.iter().take(N))
            .zip(self.dilation)
            .rev()
            .fold(1, |last_len, ((s, &d), dilation)| {
                *s = dilation as isize * last_len;
                d as isize
            });

        self.kernel
            .indexed_iter()
            .filter(|(_, v)| **v != T::zero())
            .map(|(index, v)| {
                let index = index.into_dimension();
                (
                    (0..N)
                        .map(|n| index[n] as isize * strides[n])
                        .sum::<isize>(),
                    *v,
                )
            })
            .collect()

        // let first = self.kernel.as_ptr();
        // self.kernel
        //     .iter()
        //     .filter(|v| **v != T::zero())
        //     .map(|v| (unsafe { (v as *const T).offset_from(first) }, *v))
        //     .collect()
    }
}

impl<S: RawData, const N: usize> From<ArrayBase<S, Dim<[Ix; N]>>> for KernelWithDilation<S, N> {
    fn from(kernel: ArrayBase<S, Dim<[Ix; N]>>) -> Self {
        Self {
            kernel,
            dilation: [1; N],
        }
    }
}

pub trait IntoDilation<const N: usize> {
    fn into_dilation(self) -> [usize; N];
}

impl<const N: usize> IntoDilation<N> for usize {
    #[inline]
    fn into_dilation(self) -> [usize; N] {
        [self; N]
    }
}

impl<const N: usize> IntoDilation<N> for [usize; N] {
    #[inline]
    fn into_dilation(self) -> [usize; N] {
        self
    }
}

pub trait WithDilation<S: RawData, const N: usize> {
    fn with_dilation(self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N>;
}

impl<S: RawData, const N: usize> WithDilation<S, N> for ArrayBase<S, Dim<[Ix; N]>> {
    #[inline]
    fn with_dilation(self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N> {
        KernelWithDilation {
            kernel: self,
            dilation: dilation.into_dilation(),
        }
    }
}

pub trait IntoKernelWithDilation<S: RawData, const N: usize> {
    fn into_kernel_with_dilation(self) -> KernelWithDilation<S, N>;
}

impl<S: RawData, const N: usize> IntoKernelWithDilation<S, N> for ArrayBase<S, Dim<[Ix; N]>> {
    #[inline]
    fn into_kernel_with_dilation(self) -> KernelWithDilation<S, N> {
        self.with_dilation(1)
    }
}

impl<S: RawData, const N: usize> IntoKernelWithDilation<S, N> for KernelWithDilation<S, N> {
    #[inline]
    fn into_kernel_with_dilation(self) -> KernelWithDilation<S, N> {
        self
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn check_trait_impl() {
        fn conv_example<S: RawData, const N: usize>(kernel: impl IntoKernelWithDilation<S, N>) {
            let _ = kernel.into_kernel_with_dilation();
        }

        let kernel = array![1, 0, 1];

        conv_example(kernel);

        let kernel = array![1, 0, 1];

        conv_example(kernel.with_dilation(2));

        let kernel = array![[1, 0, 1], [0, 1, 0]];

        conv_example(kernel.with_dilation([1, 2]));
    }

    #[test]
    fn check_ndarray_strides() {
        // let arr = array![[1, 1, 1], [1, 1, 1]];
        // dbg!(&arr);

        // dbg!(arr.with_dilation(2).gen_offset_list(arr.shape()));

        // let arr = array![[[1, 1, 1], [1, 1, 1]]];
        // dbg!(&arr);
    }
}
