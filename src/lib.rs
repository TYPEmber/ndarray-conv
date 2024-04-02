// mod conv_2d;
mod conv;
mod dilation;
mod padding;

// pub use conv_2d::fft::Conv2DFftExt;
// pub use conv_2d::Conv2DExt;

pub use conv::ConvExt;
use conv::ExplicitConv;
pub use padding::ExplicitPadding;

use ndarray::{ArrayBase, Dim, Dimension, IntoDimension, Ix};

// pub trait IntoExplicit<const N: usize> {
//     fn into_explicit(self) -> [usize; N];
// }

// impl<const N: usize> IntoExplicit<N> for usize {
//     #[inline]
//     fn into_explicit(self) -> [usize; N] {
//         [self; N]
//     }
// }

// impl<const N: usize> IntoExplicit<N> for [usize; N] {
//     #[inline]
//     fn into_explicit(self) -> [usize; N] {
//         self
//     }
// }

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

// impl<const N: usize> ConvMode<N> {
//     pub(crate) fn unfold<S>(self, kernel: &ArrayBase<S, Dim<[Ix; N]>>) -> ExplicitConv<N>
//     where
//         S: ndarray::RawData,
//         Dim<[Ix; N]>: Dimension,
//         [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
//     {
//         let kernel_dim = unsafe {
//             (kernel.shape().as_ptr() as *const [usize; N])
//                 .as_ref()
//                 .unwrap()
//         };

//         match self {
//             ConvMode::Full => ExplicitConv {
//                 padding: kernel_dim.map(|kernel| [kernel - 1; 2]),
//                 strides: [1; N],
//             },
//             ConvMode::Same => ExplicitConv {
//                 padding: kernel_dim.map(|k_size: usize| {
//                     if k_size % 2 == 0 {
//                         [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
//                     } else {
//                         [(k_size - 1) / 2; 2]
//                     }
//                 }),
//                 strides: [1; N],
//             },
//             ConvMode::Valid => ExplicitConv {
//                 padding: [[0; 2]; N],
//                 strides: [1; N],
//             },
//             ConvMode::Custom { padding, strides } => ExplicitConv {
//                 padding: padding.map(|pad| [pad; 2]),
//                 strides,
//             },
//             ConvMode::Explicit { padding, strides } => ExplicitConv { padding, strides },
//         }
//     }
// }
