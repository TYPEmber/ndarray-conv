mod conv;
mod conv_fft;
mod dilation;
mod padding;

pub use conv::ConvExt;
pub use padding::ExplicitPadding;

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
