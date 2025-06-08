//! Provides functionality for kernel dilation.

use ndarray::{
    ArrayBase, Data, Dim, Dimension, IntoDimension, Ix, RawData, SliceArg, SliceInfo, SliceInfoElem,
};

/// Represents a kernel along with its dilation factors for each dimension.
pub struct KernelWithDilation<'a, S: RawData, const N: usize> {
    pub(crate) kernel: &'a ArrayBase<S, Dim<[Ix; N]>>,
    pub(crate) dilation: [usize; N],
    pub(crate) reverse: bool,
}

impl<'a, S: RawData, const N: usize, T> KernelWithDilation<'a, S, N>
where
    T: num::traits::NumAssign + Copy,
    S: Data<Elem = T>,
    Dim<[Ix; N]>: Dimension,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    /// Generates a list of offsets and corresponding kernel values for efficient convolution.
    ///
    /// This method calculates the offsets into the input array that need to be accessed
    /// during the convolution operation, taking into account the kernel's dilation.
    /// It filters out elements where the kernel value is zero to optimize the computation.
    ///
    /// # Arguments
    ///
    /// * `pds_strides`: The strides of the padded input array.
    ///
    /// # Returns
    /// A `Vec` of tuples, where each tuple contains an offset and the corresponding kernel value.
    pub fn gen_offset_list(&self, pds_strides: &[isize]) -> Vec<(isize, T)> {
        let buffer_slice = self.kernel.slice(unsafe {
            SliceInfo::new(std::array::from_fn(|i| SliceInfoElem::Slice {
                start: 0,
                end: Some(self.kernel.raw_dim()[i] as isize),
                step: if self.reverse { -1 } else { 1 },
            }))
            .unwrap()
        });

        let strides: [isize; N] =
            std::array::from_fn(|i| self.dilation[i] as isize * pds_strides[i]);

        buffer_slice
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
    }
}

/// Trait for converting a value into a dilation array.
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

/// Trait for adding dilation information to a kernel.
pub trait WithDilation<S: RawData, const N: usize> {
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N>;
}

impl<S: RawData, const N: usize> WithDilation<S, N> for ArrayBase<S, Dim<[Ix; N]>> {
    #[inline]
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<S, N> {
        KernelWithDilation {
            kernel: self,
            dilation: dilation.into_dilation(),
            reverse: true,
        }
    }
}

pub trait ReverseKernel<'a, S: RawData, const N: usize> {
    fn reverse(self) -> KernelWithDilation<'a, S, N>;
    fn no_reverse(self) -> KernelWithDilation<'a, S, N>;
}

impl<'a, S: RawData, K, const N: usize> ReverseKernel<'a, S, N> for K
where
    K: IntoKernelWithDilation<'a, S, N>,
{
    #[inline]
    fn reverse(self) -> KernelWithDilation<'a, S, N> {
        let mut kwd = self.into_kernel_with_dilation();

        kwd.reverse = true;

        kwd
    }

    #[inline]
    fn no_reverse(self) -> KernelWithDilation<'a, S, N> {
        let mut kwd = self.into_kernel_with_dilation();

        kwd.reverse = false;

        kwd
    }
}

/// Trait for converting a reference to a `KernelWithDilation`.
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
        // #[deprecated(since = "0.4.2", note = "test")]
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

        // for convolution (default)
        conv_example(&kernel);
        // for convolution (explicit)
        conv_example(kernel.reverse());
        // for cross-correlation
        conv_example(kernel.with_dilation(2).no_reverse());
    }
}
