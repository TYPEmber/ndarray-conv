use crate::*;

#[derive(Debug)]
pub(crate) struct ExplicitConv<const N: usize> {
    pub pad: [[usize; 2]; N],
    pub stride: [usize; N],
}

#[derive(Debug)]
pub(crate) struct ExplictPadding<const N: usize, T: num::traits::NumAssign + Copy>(
    [[BorderType<T>; 2]; N],
);

impl<const N: usize> ConvType<N> {
    pub(crate) fn unfold(self, kernel_size: &[usize; N]) -> ExplicitConv<N> {
        match self {
            ConvType::Full => ExplicitConv {
                pad: kernel_size.map(|kernel| [kernel - 1; 2]),
                stride: std::array::from_fn(|_| 1),
            },
            ConvType::Same => {
                let split = |k_size: usize| {
                    if k_size % 2 == 0 {
                        [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
                    } else {
                        [(k_size - 1) / 2; 2]
                    }
                };

                ExplicitConv {
                    pad: kernel_size.map(split),
                    stride: std::array::from_fn(|_| 1),
                }
            }
            ConvType::Valid => {
                let (pad_hs, pad_ws) = ([0; 2], [0; 2]);
                let (stride_h, stride_w) = (1, 1);

                ExplicitConv {
                    pad: std::array::from_fn(|_| [0; 2]),
                    stride: std::array::from_fn(|_| 1),
                }
            }
            ConvType::Custom(pads, strides) => ExplicitConv {
                pad: pads.map(|pad| [pad; 2]),
                stride: strides,
            },
            ConvType::Explicit(pad, stride) => ExplicitConv { pad, stride },
        }
    }
}

impl<const N: usize, T: num::traits::NumAssign + Copy> PaddingMode<N, T> {
    pub(crate) fn unfold(self) -> ExplictPadding<N, T> {
        match self {
            PaddingMode::Zeros => ExplictPadding([[BorderType::Zeros; 2]; N]),
            PaddingMode::Const(num) => ExplictPadding([[BorderType::Const(num); 2]; N]),
            PaddingMode::Reflect => ExplictPadding([[BorderType::Reflect; 2]; N]),
            PaddingMode::Replicate => ExplictPadding([[BorderType::Replicate; 2]; N]),
            PaddingMode::Warp => ExplictPadding([[BorderType::Warp; 2]; N]),
            PaddingMode::Custom(borders) => ExplictPadding(borders.map(|border| [border; 2])),
            PaddingMode::Explicit(borders) => ExplictPadding(borders),
        }
    }
}
