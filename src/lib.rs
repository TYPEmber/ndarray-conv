//! `ndarray-conv` provides N-dimensional convolution operations for `ndarray` arrays.
//!
//! This crate extends the `ndarray` library with both standard and
//! FFT-accelerated convolution methods.
//!
//! # Getting Started
//!
//! To start performing convolutions, you'll interact with the following:
//!
//! 1. **Input Arrays:** Use `ndarray`'s [`Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
//!    or [`ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
//!    as your input data and convolution kernel.
//! 2. **Convolution Methods:** Call `array.conv(...)` or `array.conv_fft(...)`.
//!    These methods are added to `ArrayBase` types via the traits
//!    [`ConvExt::conv`] and [`ConvFFTExt::conv_fft`].
//! 3. **Convolution Mode:** [`ConvMode`] specifies the size of the output.
//! 4. **Padding Mode:** [`PaddingMode`] specifies how to handle array boundaries.
//!
//! # Basic Example:
//!
//! Here's a simple example of how to perform a 2D convolution using `ndarray-conv`:
//!
//! ```rust
//! use ndarray::prelude::*;
//! use ndarray_conv::{ConvExt, ConvFFTExt, ConvMode, PaddingMode};
//!
//! // Input data
//! let input = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
//!
//! // Convolution kernel
//! let kernel = array![[1, 1], [1, 1]];
//!
//! // Perform standard convolution with "same" output size and zero padding
//! let output = input.conv(
//!     &kernel,
//!     ConvMode::Same,
//!     PaddingMode::Zeros,
//! ).unwrap();
//!
//! println!("Standard Convolution Output:\n{:?}", output);
//!
//! // Perform FFT-accelerated convolution with "same" output size and zero padding
//! let output_fft = input.map(|&x| x as f32).conv_fft(
//!     &kernel.map(|&x| x as f32),
//!     ConvMode::Same,
//!     PaddingMode::Zeros,
//! ).unwrap();
//!
//! println!("FFT Convolution Output:\n{:?}", output_fft);
//! ```
//!
//! # Choosing a convolution method
//!
//!  * Use [`ConvExt::conv`] for standard convolution
//!  * Use [`ConvFFTExt::conv_fft`] for FFT accelerated convolution.
//!    FFT accelerated convolution is generally faster for larger kernels, but
//!    standard convolution may be faster for smaller kernels.
//!
//! # Key Structs, Enums and Traits
//!
//! *   [`ConvMode`]: Specifies how to determine the size of the convolution output (e.g., `Full`, `Same`, `Valid`).
//! *   [`PaddingMode`]: Specifies how to handle array boundaries (e.g., `Zeros`, `Reflect`, `Replicate`). You can also use `PaddingMode::Custom` or `PaddingMode::Explicit` to combine different [`BorderType`] strategies for each dimension or for each side of each dimension.
//! *   [`BorderType`]: Used with [`PaddingMode`] for `Custom` and `Explicit`, specifies the padding strategy (e.g., `Zeros`, `Reflect`, `Replicate`, `Circular`).
//! *   [`ConvExt`]: The trait that adds the `conv` method, extending `ndarray` arrays with standard convolution functionality.
//! *   [`ConvFFTExt`]: The trait that adds the `conv_fft` method, extending `ndarray` arrays with FFT-accelerated convolution functionality.

mod conv;
mod conv_fft;
mod dilation;
mod padding;

pub(crate) use padding::ExplicitPadding;

pub use conv::ConvExt;
pub use conv_fft::{ConvFFTExt, Processor as FftProcessor};
pub use dilation::{WithDilation, ReverseKernel};

/// Specifies the convolution mode, which determines the output size.
#[derive(Debug, Clone, Copy)]
pub enum ConvMode<const N: usize> {
    /// The output has the largest size, including all positions where
    /// the kernel and input overlap at least partially.
    Full,
    /// The output has the same size as the input.
    Same,
    /// The output has the smallest size, including only positions
    /// where the kernel and input fully overlap.
    Valid,
    /// Specifies custom padding and strides.
    Custom {
        /// The padding to use for each dimension.
        padding: [usize; N],
        /// The strides to use for each dimension.
        strides: [usize; N],
    },
    /// Specifies explicit padding and strides.
    Explicit {
        /// The padding to use for each side of each dimension.
        padding: [[usize; 2]; N],
        /// The strides to use for each dimension.
        strides: [usize; N],
    },
}
/// Specifies the padding mode, which determines how to handle borders.
///
/// The padding mode can be either a single `BorderType` applied on all sides
/// or a custom tuple of two `BorderTypes` for each dimension or a `BorderType`
/// for each side of each dimension.
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode<const N: usize, T: num::traits::NumAssign + Copy> {
    /// Pads with zeros.
    Zeros,
    /// Pads with a constant value.
    Const(T),
    /// Reflects the input at the borders.
    Reflect,
    /// Replicates the edge values.
    Replicate,
    /// Treats the input as a circular buffer.
    Circular,
    /// Specifies a different `BorderType` for each dimension.
    Custom([BorderType<T>; N]),
    /// Specifies a different `BorderType` for each side of each dimension.
    Explicit([[BorderType<T>; 2]; N]),
}

/// Used with [`PaddingMode`]. Specifies the padding mode for a single dimension
/// or a single side of a dimension.
#[derive(Debug, Clone, Copy)]
pub enum BorderType<T: num::traits::NumAssign + Copy> {
    /// Pads with zeros.
    Zeros,
    /// Pads with a constant value.
    Const(T),
    /// Reflects the input at the borders.
    Reflect,
    /// Replicates the edge values.
    Replicate,
    /// Treats the input as a circular buffer.
    Circular,
}

use thiserror::Error;

/// Error type for convolution operations.
#[derive(Error, Debug)]
pub enum Error<const N: usize> {
    /// Indicates that the input data array has a dimension with zero size.
    #[error("Data shape shouldn't have ZERO. {0:?}")]
    DataShape(ndarray::Dim<[ndarray::Ix; N]>),
    /// Indicates that the kernel array has a dimension with zero size.
    #[error("Kernel shape shouldn't have ZERO. {0:?}")]
    KernelShape(ndarray::Dim<[ndarray::Ix; N]>),
    /// Indicates that the shape of the kernel with dilation is not compatible with the chosen `ConvMode`.
    #[error("ConvMode {0:?} does not match KernelWithDilation Size {1:?}")]
    MismatchShape(ConvMode<N>, [ndarray::Ix; N]),
}
