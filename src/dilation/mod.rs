use ndarray::{ArrayBase, Data, Dim, Dimension, IntoDimension, Ix, RawData};

pub struct KernelWithDilation<'a, S: RawData, const N: usize> {
    pub kernel: &'a ArrayBase<S, Dim<[Ix; N]>>,
    pub dilation: [usize; N],
}

impl<'a, S: RawData, const N: usize, T> KernelWithDilation<'a, S, N>
where
    T: num::traits::NumAssign + Copy,
    S: Data<Elem = T>,
    Dim<[Ix; N]>: Dimension,
{
    pub fn gen_offset_list(&self, pds_strides: &[isize]) -> Vec<(isize, T)> {
        let strides: [isize; N] =
            std::array::from_fn(|i| self.dilation[i] as isize * pds_strides[i]);

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

impl<'a, S: RawData, const N: usize> From<&'a ArrayBase<S, Dim<[Ix; N]>>>
    for KernelWithDilation<'a, S, N>
{
    fn from(kernel: &'a ArrayBase<S, Dim<[Ix; N]>>) -> Self {
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
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N>;
}

impl<S: RawData, const N: usize> WithDilation<S, N> for ArrayBase<S, Dim<[Ix; N]>> {
    #[inline]
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N> {
        KernelWithDilation {
            kernel: self,
            dilation: dilation.into_dilation(),
        }
    }
}

pub trait IntoKernelWithDilation<'a, S: RawData, const N: usize> {
    fn into_kernel_with_dilation(self) -> KernelWithDilation<'a, S, N>;
}

impl<'a, S: RawData, const N: usize> IntoKernelWithDilation<'a, S, N>
    for &'a ArrayBase<S, Dim<[Ix; N]>>
{
    #[inline]
    fn into_kernel_with_dilation(self) -> KernelWithDilation<'a, S, N> {
        self.with_dilation(1)
    }
}

impl<'a, S: RawData, const N: usize> IntoKernelWithDilation<'a, S, N>
    for KernelWithDilation<'a, S, N>
{
    #[inline]
    fn into_kernel_with_dilation(self) -> KernelWithDilation<'a, S, N> {
        self
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn check_trait_impl() {
        fn conv_example<'a, S: RawData + 'a, const N: usize>(
            kernel: impl IntoKernelWithDilation<'a, S, N>,
        ) {
            let _ = kernel.into_kernel_with_dilation();
        }

        let kernel = array![1, 0, 1];

        conv_example(&kernel);

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
