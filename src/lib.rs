mod conv_2d;

pub use conv_2d::fft::Conv2DFftExt;
pub use conv_2d::Conv2DExt;

#[derive(Debug, Clone, Copy)]
pub enum Padding<const N: usize> {
    Full,
    Same,
    Valid,
    // (pad, stride)
    Custom([usize; N], [usize; N]),
    // (pad, stride)
    Explicit([[usize; 2]; N], [usize; N]),
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

#[derive(Debug)]
pub(crate) struct ExplicitPadding<const N: usize> {
    pub pad: [[usize; 2]; N],
    pub stride: [usize; N],
}

#[derive(Debug)]
pub(crate) struct ExplictMode<const N: usize, T: num::traits::NumAssign + Copy>(
    pub [[BorderType<T>; 2]; N],
);

impl<const N: usize> Padding<N> {
    pub(crate) fn unfold(self, kernel_size: &[usize; N]) -> ExplicitPadding<N> {
        match self {
            Padding::Full => ExplicitPadding {
                pad: kernel_size.map(|kernel| [kernel - 1; 2]),
                stride: std::array::from_fn(|_| 1),
            },
            Padding::Same => {
                let split = |k_size: usize| {
                    if k_size % 2 == 0 {
                        [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
                    } else {
                        [(k_size - 1) / 2; 2]
                    }
                };

                ExplicitPadding {
                    pad: kernel_size.map(split),
                    stride: std::array::from_fn(|_| 1),
                }
            }
            Padding::Valid => ExplicitPadding {
                pad: std::array::from_fn(|_| [0; 2]),
                stride: std::array::from_fn(|_| 1),
            },
            Padding::Custom(pads, strides) => ExplicitPadding {
                pad: pads.map(|pad| [pad; 2]),
                stride: strides,
            },
            Padding::Explicit(pad, stride) => ExplicitPadding { pad, stride },
        }
    }
}

impl<const N: usize, T: num::traits::NumAssign + Copy> PaddingMode<N, T> {
    pub(crate) fn unfold(self) -> ExplictMode<N, T> {
        match self {
            PaddingMode::Zeros => ExplictMode([[BorderType::Zeros; 2]; N]),
            PaddingMode::Const(num) => ExplictMode([[BorderType::Const(num); 2]; N]),
            PaddingMode::Reflect => ExplictMode([[BorderType::Reflect; 2]; N]),
            PaddingMode::Replicate => ExplictMode([[BorderType::Replicate; 2]; N]),
            PaddingMode::Circular => ExplictMode([[BorderType::Circular; 2]; N]),
            PaddingMode::Custom(borders) => ExplictMode(borders.map(|border| [border; 2])),
            PaddingMode::Explicit(borders) => ExplictMode(borders),
        }
    }
}