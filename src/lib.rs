mod conv;
mod conv_fft;
mod dilation;
mod padding;

pub(crate) use padding::ExplicitPadding;

pub use conv::ConvExt;
pub use conv_fft::ConvFFTExt;

#[derive(Debug, Clone, Copy)]
pub enum ConvMode<const N: usize> {
    Full,
    Same,
    Valid,
    // (pad, stride)
    Custom {
        padding: [usize; N],
        strides: [usize; N],
    },
    // (pad, stride)
    Explicit {
        padding: [[usize; 2]; N],
        strides: [usize; N],
    },
}

// padding mode. It can be either a single BorderType applied on all sides or a custom tuple of two BorderTypes for (H, W), respectively.
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode<const N: usize, T: num::traits::NumAssign + Copy> {
    Zeros,
    Const(T),
    Reflect,
    Replicate,
    Circular,
    Custom([BorderType<T>; N]),
    Explicit([[BorderType<T>; 2]; N]),
}

// padding mode for single dim
#[derive(Debug, Clone, Copy)]
pub enum BorderType<T: num::traits::NumAssign + Copy> {
    Zeros,
    Const(T),
    Reflect,
    Replicate,
    Circular,
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<const N: usize> {
    #[error("Data shape shouldn't have ZERO. {0:?}")]
    DataShape(ndarray::Dim<[ndarray::Ix; N]>),
    #[error("Kernel shape shouldn't have ZERO. {0:?}")]
    KernelShape(ndarray::Dim<[ndarray::Ix; N]>),
    #[error("ConvMode {0:?} does not match KernelWithDilation Size {1:?}")]
    MismatchShape(ConvMode<N>, [ndarray::Ix; N]),
}
