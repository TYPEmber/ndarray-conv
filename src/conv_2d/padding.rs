use crate::{BorderType, ConvType, PaddingMode};
use ndarray::{prelude::*, StrideShape};
use num::traits::NumAssign;

impl ConvType<2> {
    pub fn unfold(self, kernel_size: &[usize; 2]) -> Self {
        match self {
            ConvType::Full => {
                let (stride_h, stride_w) = (1, 1);
                let [pad_h, pad_w] = [kernel_size[0] - 1, kernel_size[1] - 1];
                ConvType::Explicit([[pad_h; 2], [pad_w; 2]], [stride_h, stride_w])
            }
            ConvType::Same => {
                let (stride_h, stride_w) = (1, 1);
                let (kernel_h, kernel_w) = (kernel_size[0], kernel_size[1]);

                let split = |k_size: usize| {
                    if k_size % 2 == 0 {
                        [(k_size - 1) / 2 + 1, (k_size - 1) / 2]
                    } else {
                        [(k_size - 1) / 2; 2]
                    }
                };

                ConvType::Explicit([split(kernel_h), split(kernel_w)], [stride_h, stride_w])
            }
            ConvType::Valid => {
                let (pad_hs, pad_ws) = ([0; 2], [0; 2]);
                let (stride_h, stride_w) = (1, 1);
                ConvType::Explicit([pad_hs, pad_ws], [stride_h, stride_w])
            }
            ConvType::Custom(pads, strides) => ConvType::Explicit([pads, pads], strides),
            ConvType::Explicit(_, _) => self,
        }
    }
}

// pub fn get_size
pub fn get_size(
    input_size: &[usize; 2],
    kernel_size: &[usize; 2],
    conv_type: ConvType<2>,
) -> ([usize; 2], [usize; 2]) {
    if let ConvType::Explicit(pads, strides) = conv_type.unfold(kernel_size) {
        let (input_h, input_w) = (input_size[0], input_size[1]);
        let (kernel_h, kernel_w) = (kernel_size[0], kernel_size[1]);

        let pad_input_size = [
            input_h + pads[0].iter().sum::<usize>(),
            input_w + pads[1].iter().sum::<usize>(),
        ];

        let out_size = [
            (input_h - kernel_h + pads[0].iter().sum::<usize>()) / strides[0] + 1,
            (input_w - kernel_w + pads[1].iter().sum::<usize>()) / strides[1] + 1,
        ];
        (pad_input_size, out_size)

    } else {
        unreachable!()
    }

    // let (pad, stride) = match conv_type {
    //     ConvType::Full => {
    //         let (pad_h, pad_w) = ([kernel_h - 1; 2], [kernel_w - 1; 2]);
    //         let (stride_h, stride_w) = (1, 1);
    //         ([pad_h, pad_w], [stride_h, stride_w])
    //     }
    //     ConvType::Same => {
    //         let (stride_h, stride_w) = (1, 1);
    //         let pad_h = if (kernel_h - 1) % 2 == 0 {
    //             [(kernel_h - 1) / 2; 2]
    //         } else {
    //             [(kernel_h - 1) / 2 + 1, (kernel_h - 1) / 2]
    //         };
    //         let pad_w = if (kernel_w - 1) % 2 == 0 {
    //             [(kernel_w - 1) / 2; 2]
    //         } else {
    //             [(kernel_w - 1) / 2 + 1, (kernel_w - 1) / 2]
    //         };
    //         ([pad_h, pad_w], [stride_h, stride_w])
    //     }
    //     ConvType::Valid => {
    //         let (pad_h, pad_w) = ([0; 2], [0; 2]);
    //         let (stride_h, stride_w) = (1, 1);
    //         ([pad_h, pad_w], [stride_h, stride_w])
    //     }
    //     ConvType::Custom(pad, stride) => ([[pad[0]; 2], [pad[1]; 2]], stride),
    //     ConvType::Explicit(pad, stride) => (pad, stride),
    // };
}

pub fn pad<S, T>(
    data: &ArrayBase<S, Ix2>,
    padding: &[[usize; 2]; 2],
    padding_size: &[[usize; 2]; 2],
    pad_input_size: &[usize; 2],
    padding_mode: PaddingMode<2, T>,
) -> Array2<T>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let (pad_input_h, pad_input_w) = (pad_input_size[0], pad_input_size[1]);
    let pad_input = Array2::zeros((pad_input_h, pad_input_w));
    pad_inner(
        data,
        padding,
        &[pad_input_h, pad_input_w],
        pad_input,
        padding_mode,
    )
}

fn pad_inner<S, T>(
    data: &ArrayBase<S, Ix2>,
    padding_size: &[[usize; 2]; 2],
    pad_input_size: &[usize; 2],
    mut pad_input: Array2<T>,
    padding_mode: PaddingMode<2, T>,
) -> Array2<T>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let (input_h, input_w) = (data.shape()[0], data.shape()[1]);
    let [padding_h, padding_w] = *padding_size;
    let (pad_input_h, pad_input_w) = (pad_input_size[0], pad_input_size[1]);

    // pad input
    // let mut pad_input = Array2::zeros((pad_input_h, pad_input_w));
    let mut sub_pad_input = pad_input.slice_mut(s!(
        padding_h[0]..input_h + padding_h[0],
        padding_w[0]..input_w + padding_w[0]
    ));
    sub_pad_input.assign(data);

    // dbg!(& pad_input);

    let pair_padding_mode = padding_mode.unfold();

    // let mut pad_input = Array2::zeros((pad_input_h, pad_input_w));

    // match padding_mode {
    //     PaddingMode::Zeros => Array2::zeros((pad_input_h, pad_input_w)),
    //     PaddingMode::Const(num) => Array2::from_elem((pad_input_h, pad_input_w), num),
    //     PaddingMode::Reflect => Array2::zeros((1, 1)),
    //     PaddingMode::Replicate => Array2::zeros((1, 1)),
    //     PaddingMode::Warp => Array2::zeros((1, 1)),
    //     PaddingMode::Custom([row_border_type, col_border_type]) => Array2::zeros((1, 1)),
    // }
    pad_inner_row(
        &mut pad_input,
        input_h,
        input_w,
        padding_size,
        &pair_padding_mode,
    );
    pad_inner_col(
        &mut pad_input,
        input_h,
        input_w,
        padding_size,
        &pair_padding_mode,
    );
    // Array2::zeros((1, 1))
    pad_input
}

fn pad_inner_row<T>(
    pad_input: &mut Array2<T>,
    input_h: usize,
    input_w: usize,
    padding_size: &[[usize; 2]; 2],
    border_type: &PaddingMode<2, T>,
) where
    T: Copy + NumAssign + std::fmt::Debug,
{
    let [padding_h, padding_w] = *padding_size;

    if let PaddingMode::Custom([row_border_type, _]) = border_type {
        match row_border_type {
            BorderType::Const(num) => {
                // left padding
                pad_input
                    .slice_mut(s!(padding_h[0]..input_h + padding_h[0], 0..padding_w[0]))
                    .assign(&Array2::from_elem((input_h, padding_w[0]), *num));
                // right padding
                pad_input
                    .slice_mut(s!(
                        padding_h[0]..input_h + padding_h[0],
                        input_w + padding_w[0]..
                    ))
                    .assign(&Array2::from_elem((input_h, padding_w[1]), *num));
            }
            BorderType::Reflect => {
                // left padding
                for i in padding_h[0]..padding_h[0] + input_h {
                    for j in 1..=padding_w[0] {
                        pad_input[[i, padding_w[0] - j]] = pad_input[[i, padding_w[0] + j]];
                    }
                }
                // right padding
                for i in padding_h[0]..padding_h[0] + input_h {
                    for j in 1..=padding_w[1] {
                        pad_input[[i, padding_w[0] + input_w - 1 + j]] =
                            pad_input[[i, padding_w[0] + input_w - 1 - j]]
                    }
                }
            }
            BorderType::Replicate => {
                // left padding
                for mut row in pad_input
                    .slice_mut(s!(
                        padding_h[0]..input_h + padding_h[0],
                        0..padding_w[0] + 1
                    ))
                    .rows_mut()
                {
                    let last_elem = *row.last().unwrap();
                    row.slice_mut(s!(..padding_w[0]))
                        .assign(&Array1::from_elem(padding_w[0], last_elem));
                }
                // right padding
                for mut row in pad_input
                    .slice_mut(s!(
                        padding_h[0]..input_h + padding_h[0],
                        input_w + padding_w[0] - 1..
                    ))
                    .rows_mut()
                {
                    let first_elem = *row.first().unwrap();
                    row.slice_mut(s!(1..))
                        .assign(&Array1::from_elem(padding_w[0], first_elem));
                }
            }
            BorderType::Warp => unsafe {
                // left padding
                let left_pad = pad_input.slice(s!(
                    padding_h[0]..input_h + padding_h[0],
                    input_w..padding_w[0] + input_w
                ));

                (&pad_input.slice(s!(padding_h[0]..input_h + padding_h[0], 0..padding_w[0]))
                    as *const ArrayView2<T> as *mut ArrayViewMut2<T>)
                    .as_mut()
                    .unwrap()
                    .assign(&left_pad);

                // right padding
                let right_pad = pad_input.slice(s!(
                    padding_h[0]..input_h + padding_h[0],
                    padding_w[0]..padding_w[0] + padding_w[1]
                ));

                (&pad_input.slice(s!(
                    padding_h[0]..input_h + padding_h[0],
                    input_w + padding_w[0]..
                )) as *const ArrayView2<T> as *mut ArrayViewMut2<T>)
                    .as_mut()
                    .unwrap()
                    .assign(&right_pad);
            },
            _ => {}
        }
    }
}

fn pad_inner_col<T>(
    pad_input: &mut Array2<T>,
    input_h: usize,
    input_w: usize,
    padding_size: &[[usize; 2]; 2],
    border_type: &PaddingMode<2, T>,
) where
    T: Copy + NumAssign + std::fmt::Debug,
{
    let [padding_h, padding_w] = *padding_size;

    if let PaddingMode::Custom([_, col_border_type]) = border_type {
        match col_border_type {
            BorderType::Const(num) => {
                // top padding
                pad_input
                    .slice_mut(s!(..padding_h[0], ..))
                    .assign(&Array2::from_elem(
                        (padding_h[0], input_w + padding_w[0] + padding_w[1]),
                        *num,
                    ));
                // bottom padding
                pad_input
                    .slice_mut(s!(padding_h[0] + input_h.., ..))
                    .assign(&Array2::from_elem(
                        (padding_h[1], input_w + padding_w[0] + padding_w[1]),
                        *num,
                    ));
            }
            BorderType::Reflect => unsafe {
                // top padding
                for i in 1..=padding_h[0] {
                    let row_pad = pad_input.row(padding_h[0] + i);

                    (&pad_input.row(padding_h[0] - i) as *const ArrayView1<T>
                        as *mut ArrayViewMut1<T>)
                        .as_mut()
                        .unwrap()
                        .assign(&row_pad);
                }
                // bottom padding
                for i in 1..=padding_h[1] {
                    let row_pad = pad_input.row(padding_h[0] + input_h - 1 - i);

                    (&pad_input.row(padding_h[0] + input_h - 1 + i) as *const ArrayView1<T>
                        as *mut ArrayViewMut1<T>)
                        .as_mut()
                        .unwrap()
                        .assign(&row_pad);
                }
            },
            BorderType::Replicate => unsafe {
                let first_row = pad_input.row(padding_h[0]);
                let last_row = pad_input.row(input_h + padding_h[0] - 1);
                // top padding
                for i in 0..padding_h[0] {
                    (&pad_input.row(i) as *const ArrayView1<T> as *mut ArrayViewMut1<T>)
                        .as_mut()
                        .unwrap()
                        .assign(&first_row);
                }
                // bottom padding
                for i in input_h + padding_h[0] - 1..input_h + padding_h[0] + padding_h[1] {
                    (&pad_input.row(i) as *const ArrayView1<T> as *mut ArrayViewMut1<T>)
                        .as_mut()
                        .unwrap()
                        .assign(&last_row);
                }
            },
            BorderType::Warp => unsafe {
                // top padding
                let top_pad = pad_input.slice(s!(input_h..input_h + padding_h[0], ..));

                (&pad_input.slice(s!(..padding_h[0], ..)) as *const ArrayView2<T>
                    as *mut ArrayViewMut2<T>)
                    .as_mut()
                    .unwrap()
                    .assign(&top_pad);

                // bottom padding
                let bottom_pad = pad_input.slice(s!(padding_h[0]..padding_h[0] + padding_h[1], ..));

                (&pad_input.slice(s!(padding_h[0] + input_h.., ..)) as *const ArrayView2<T>
                    as *mut ArrayViewMut2<T>)
                    .as_mut()
                    .unwrap()
                    .assign(&bottom_pad);
            },
            _ => {}
        }
    }

    // match border_type {
    //     PaddingMode::Custom([row_border_type, col_border_type]) => {

    //     }
    //     _=> {}
    // }
    // match
    // Array2::zeros((1, 1))
}

impl<T: NumAssign + Copy> PaddingMode<2, T> {
    pub fn unfold(self) -> Self {
        match self {
            PaddingMode::Zeros => PaddingMode::Custom([BorderType::Zeros, BorderType::Zeros]),
            PaddingMode::Const(num) => {
                PaddingMode::Custom([BorderType::Const(num), BorderType::Const(num)])
            }
            PaddingMode::Reflect => PaddingMode::Custom([BorderType::Reflect, BorderType::Reflect]),
            PaddingMode::Replicate => {
                PaddingMode::Custom([BorderType::Replicate, BorderType::Replicate])
            }
            PaddingMode::Warp => PaddingMode::Custom([BorderType::Warp; 2]),
            PaddingMode::Custom(_) => self,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn padding_mode_unfold() {}
}
