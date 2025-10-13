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
///
/// Dilation is a parameter that controls the spacing between kernel elements
/// during convolution. A dilation of 1 means no spacing (standard convolution),
/// while larger values insert gaps between kernel elements.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use ndarray_conv::{WithDilation, ConvExt, ConvMode, PaddingMode};
///
/// let input = array![1, 2, 3, 4, 5];
/// let kernel = array![1, 1, 1];
///
/// // Standard convolution (dilation = 1)
/// let result1 = input.conv(&kernel, ConvMode::Same, PaddingMode::Zeros).unwrap();
///
/// // Dilated convolution (dilation = 2)
/// let result2 = input.conv(kernel.with_dilation(2), ConvMode::Same, PaddingMode::Zeros).unwrap();
/// ```
pub trait WithDilation<S: RawData, const N: usize> {
    /// Adds dilation information to the kernel.
    ///
    /// # Arguments
    ///
    /// * `dilation`: The dilation factor(s). Can be a single value (applied to all dimensions)
    ///   or an array of values (one per dimension).
    ///
    /// # Returns
    ///
    /// A `KernelWithDilation` instance containing the kernel and dilation information.
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<'_, S, N>;
}

impl<S: RawData, const N: usize> WithDilation<S, N> for ArrayBase<S, Dim<[Ix; N]>> {
    #[inline]
    fn with_dilation(&self, dilation: impl IntoDilation<N>) -> KernelWithDilation<'_, S, N> {
        KernelWithDilation {
            kernel: self,
            dilation: dilation.into_dilation(),
            reverse: true,
        }
    }
}

/// Trait for controlling kernel reversal behavior in convolution operations.
///
/// In standard convolution, the kernel is reversed (flipped) along all axes.
/// This trait allows you to control whether the kernel should be reversed or not.
///
/// # Convolution vs Cross-Correlation
///
/// * **Convolution** (default, `reverse()`): The kernel is reversed, which is the mathematical definition of convolution.
/// * **Cross-correlation** (`no_reverse()`): The kernel is NOT reversed. This is commonly used in machine learning frameworks.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use ndarray_conv::{WithDilation, ReverseKernel, ConvExt, ConvMode, PaddingMode};
///
/// let input = array![1, 2, 3, 4, 5];
/// let kernel = array![1, 2, 3];
///
/// // Standard convolution (kernel is reversed)
/// let result1 = input.conv(&kernel, ConvMode::Same, PaddingMode::Zeros).unwrap();
/// // Equivalent to:
/// let result1_explicit = input.conv(kernel.reverse(), ConvMode::Same, PaddingMode::Zeros).unwrap();
///
/// // Cross-correlation (kernel is NOT reversed)
/// let result2 = input.conv(kernel.no_reverse(), ConvMode::Same, PaddingMode::Zeros).unwrap();
/// ```
pub trait ReverseKernel<'a, S: RawData, const N: usize> {
    /// Explicitly enables kernel reversal (standard convolution).
    ///
    /// This is the default behavior, so calling this method is usually not necessary.
    fn reverse(self) -> KernelWithDilation<'a, S, N>;

    /// Disables kernel reversal (cross-correlation).
    ///
    /// Use this when you want the kernel to be applied without flipping,
    /// which is common in machine learning applications.
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

    // ===== Trait Implementation Tests =====

    mod trait_implementation {
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

            // for convolution (default)
            conv_example(&kernel);
            // for convolution (explicit)
            conv_example(kernel.reverse());
            // for cross-correlation
            conv_example(kernel.with_dilation(2).no_reverse());
        }
    }

    // ===== Basic API Tests =====

    mod basic_api {
        use super::*;

        #[test]
        fn dilation_and_reverse_settings() {
            let kernel = array![1, 2, 3];

            // Test dilation is set correctly for different dimensions
            assert_eq!(kernel.with_dilation(2).dilation, [2]);
            assert_eq!(array![[1, 2]].with_dilation([2, 3]).dilation, [2, 3]);
            assert_eq!(array![[[1]]].with_dilation([1, 2, 3]).dilation, [1, 2, 3]);

            // Test reverse behavior (default is true, can be toggled)
            assert!(kernel.with_dilation(1).reverse);
            assert!(!kernel.with_dilation(1).no_reverse().reverse);
            assert!(kernel.with_dilation(1).no_reverse().reverse().reverse);
        }
    }

    // ===== Offset Generation Tests =====

    mod offset_generation {
        use super::*;

        #[test]
        fn gen_offset_1d_no_dilation() {
            let kernel = array![1.0, 2.0, 3.0];
            let kwd = kernel.with_dilation(1);

            // Stride = 1 for 1D
            let offsets = kwd.gen_offset_list(&[1]);

            // Should have 3 offsets (all kernel elements)
            assert_eq!(offsets.len(), 3);

            // With reverse=true, kernel is reversed: [3, 2, 1]
            // Offsets: [0, 1, 2] * stride[1] = [0, 1, 2]
            assert_eq!(offsets[0], (0, 3.0));
            assert_eq!(offsets[1], (1, 2.0));
            assert_eq!(offsets[2], (2, 1.0));
        }

        #[test]
        fn gen_offset_1d_with_dilation() {
            let kernel = array![1.0, 2.0, 3.0];
            let kwd = kernel.with_dilation(2);

            // Stride = 1, but dilation = 2
            let offsets = kwd.gen_offset_list(&[1]);

            assert_eq!(offsets.len(), 3);

            // Effective kernel: [1, 0, 2, 0, 3]
            // With reverse, indices with dilation: [0*2, 1*2, 2*2] = [0, 2, 4]
            // But reversed: [3, 2, 1] at positions [0, 2, 4]
            assert_eq!(offsets[0], (0, 3.0));
            assert_eq!(offsets[1], (2, 2.0));
            assert_eq!(offsets[2], (4, 1.0));
        }

        #[test]
        fn gen_offset_1d_no_reverse() {
            let kernel = array![1.0, 2.0, 3.0];
            let kwd = kernel.with_dilation(2).no_reverse();

            let offsets = kwd.gen_offset_list(&[1]);

            assert_eq!(offsets.len(), 3);

            // No reverse: [1, 2, 3] at positions [0, 2, 4]
            assert_eq!(offsets[0], (0, 1.0));
            assert_eq!(offsets[1], (2, 2.0));
            assert_eq!(offsets[2], (4, 3.0));
        }

        #[test]
        fn gen_offset_2d_no_dilation() {
            let kernel = array![[1.0, 2.0], [3.0, 4.0]];
            let kwd = kernel.with_dilation(1);

            // Strides for 2D: [row_stride, col_stride]
            let offsets = kwd.gen_offset_list(&[10, 1]);

            assert_eq!(offsets.len(), 4);

            // With reverse, kernel becomes [[4, 3], [2, 1]]
            // Flattened in row-major order with reversed indices:
            // (0,0)=4 at offset 0, (0,1)=3 at offset 1, (1,0)=2 at offset 10, (1,1)=1 at offset 11
            assert_eq!(offsets[0], (0, 4.0));
            assert_eq!(offsets[1], (1, 3.0));
            assert_eq!(offsets[2], (10, 2.0));
            assert_eq!(offsets[3], (11, 1.0));
        }

        #[test]
        fn gen_offset_2d_with_dilation() {
            let kernel = array![[1.0, 2.0], [3.0, 4.0]];
            let kwd = kernel.with_dilation([2, 3]);

            let offsets = kwd.gen_offset_list(&[10, 1]);

            assert_eq!(offsets.len(), 4);

            // Dilation [2, 3] means:
            // - row spacing = 2 (kernel rows are 0 and 2*10=20 apart)
            // - col spacing = 3 (kernel cols are 0 and 3*1=3 apart)
            // With reverse, kernel [[4,3],[2,1]] at effective positions:
            // (0,0)=4 at 0, (0,3)=3 at 3, (2,0)=2 at 20, (2,3)=1 at 23
            assert_eq!(offsets[0], (0, 4.0));
            assert_eq!(offsets[1], (3, 3.0));
            assert_eq!(offsets[2], (20, 2.0));
            assert_eq!(offsets[3], (23, 1.0));
        }

        #[test]
        fn gen_offset_filters_zeros() {
            let kernel = array![1.0, 0.0, 2.0, 0.0, 3.0];
            let kwd = kernel.with_dilation(1);

            let offsets = kwd.gen_offset_list(&[1]);

            // Should only have 3 offsets (non-zero elements)
            assert_eq!(offsets.len(), 3);
        }
    }

    // ===== Edge Cases =====

    mod edge_cases {
        use super::*;

        #[test]
        fn single_element_kernel() {
            let kernel = array![42.0];
            let kwd = kernel.with_dilation(5);

            assert_eq!(kwd.dilation, [5]);

            let offsets = kwd.gen_offset_list(&[1]);
            assert_eq!(offsets.len(), 1);
            assert_eq!(offsets[0], (0, 42.0));
        }

        #[test]
        fn all_zeros_kernel() {
            let kernel = array![0.0, 0.0, 0.0];
            let kwd = kernel.with_dilation(2);

            let offsets = kwd.gen_offset_list(&[1]);
            // Should filter out all zeros
            assert_eq!(offsets.len(), 0);
        }

        #[test]
        fn large_dilation_value() {
            let kernel = array![1, 2];
            let kwd = kernel.with_dilation(100);

            assert_eq!(kwd.dilation, [100]);
            // Effective size: 2 + (2-1)*99 = 101
        }

        #[test]
        fn asymmetric_2d_dilation() {
            let kernel = array![[1, 2, 3], [4, 5, 6]];
            let kwd = kernel.with_dilation([1, 5]);

            assert_eq!(kwd.dilation, [1, 5]);
            // dim 0: no dilation (keeps 2 rows)
            // dim 1: dilation=5 (3 + (3-1)*4 = 11 effective cols)
        }
    }

    // ===== Integration Tests =====

    mod integration_with_padding {
        use super::*;

        #[test]
        fn effective_kernel_size_calculation() {
            // This tests the concept used in padding calculations
            let kernel = array![1, 2, 3];

            // No dilation
            let kwd1 = kernel.with_dilation(1);
            let effective_size_1 = kernel.len() + (kernel.len() - 1) * (kwd1.dilation[0] - 1);
            assert_eq!(effective_size_1, 3);

            // Dilation = 2
            let kwd2 = kernel.with_dilation(2);
            let effective_size_2 = kernel.len() + (kernel.len() - 1) * (kwd2.dilation[0] - 1);
            assert_eq!(effective_size_2, 5);

            // Dilation = 3
            let kwd3 = kernel.with_dilation(3);
            let effective_size_3 = kernel.len() + (kernel.len() - 1) * (kwd3.dilation[0] - 1);
            assert_eq!(effective_size_3, 7);
        }
    }
}
