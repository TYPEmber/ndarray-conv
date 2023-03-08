use crate::{BorderType, PaddingMode, PaddingSize};
use crate::{ExplicitPadding, ExplictMode};
use ndarray::prelude::*;
// use ndarray::{prelude::*, StrideShape};
use num::traits::NumAssign;

const N: usize = 2;

// pub fn get_size
pub(crate) fn get_size(
    input_size: &[usize; N],
    kernel_size: &[usize; N],
    conv_type: &ExplicitPadding<N>,
) -> ([usize; 2], [usize; 2]) {
    // let ExplicitConv { pad, stride } = conv_type.unfold(kernel_size);
    let (input_h, input_w) = (input_size[0], input_size[1]);
    let (kernel_h, kernel_w) = (kernel_size[0], kernel_size[1]);
    let [pad_hs, pad_ws] = conv_type.pad;
    let [stride_h, stride_w] = conv_type.stride;
    // let pad_input_size = input_size
    //     .iter()
    //     .zip(pad.iter())
    //     .map(|(input, pad)| input + pad.iter().sum::<usize>());

    // let pad_input_size = [
    //     input_h + pad[0].iter().sum::<usize>(),
    //     input_w + pad[1].iter().sum::<usize>(),
    // ];
    let pad_input_size = [
        input_h + pad_hs.iter().sum::<usize>(),
        input_w + pad_ws.iter().sum::<usize>(),
    ];
    // let out_size = input_size
    //     .iter()
    //     .zip(kernel_size.iter())
    //     .zip(pad.iter())
    //     .zip(stride.iter())
    //     .map(|(((input, kernel), pad), stride)| {
    //         input - kernel + pad.iter().sum::<usize>() / stride + 1
    //     });

    let out_size = [
        (input_h - kernel_h + pad_hs.iter().sum::<usize>()) / stride_h + 1,
        (input_w - kernel_w + pad_ws.iter().sum::<usize>()) / stride_w + 1,
    ];
    (pad_input_size, out_size)

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

pub(crate) fn pad<S, T>(
    data: &ArrayBase<S, Ix2>,
    padding: &ExplicitPadding<N>,
    padding_mode: &PaddingMode<N, T>,
) -> Option<Array2<T>>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let mut buf = match *padding_mode {
        PaddingMode::Zeros => return pad_const(data, padding, T::zero()).into(),
        PaddingMode::Const(value) => return pad_const(data, padding, value).into(),
        _ => pad_const(data, padding, T::zero()),
    };

    pad_inner(data, buf.view_mut(), padding, padding_mode)?;
    buf.into()
}

#[inline]
pub(crate) fn pad_inner<'a, S, T>(
    data: &ArrayBase<S, Ix2>,
    mut buf: ArrayViewMut2<'a, T>,
    padding: &ExplicitPadding<N>,
    padding_mode: &PaddingMode<N, T>,
) -> Option<ArrayViewMut2<'a, T>>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    let padding_mode = padding_mode.unfold();
    let (input_h, input_w) = (data.shape()[0], data.shape()[1]);

    let mut buf_row_padded = pad_inner_row(
        buf.view_mut(),
        input_h,
        input_w,
        &padding.pad,
        &padding_mode.0[0],
    )?;
    pad_inner_col(
        buf_row_padded.view_mut(),
        input_h,
        input_w,
        &padding.pad,
        &padding_mode.0[1],
    )?;

    buf.into()
}

pub(crate) fn pad_const<S, T>(data: &ArrayBase<S, Ix2>, padding: &ExplicitPadding<N>, value: T) -> Array2<T>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    // let buf_shape: [usize; N] = std::array::from_fn(|i| unsafe {
    //     data.shape().get_unchecked(i) + padding.pad.get_unchecked(i).iter().sum::<usize>()
    // });
    let buf_shape = [
        data.shape()[0] + padding.pad[0].iter().sum::<usize>(),
        data.shape()[1] + padding.pad[1].iter().sum::<usize>(),
    ];
    let mut buf = Array2::from_elem(buf_shape, value);

   pad_const_inner(data, buf.view_mut(), padding);

   buf
}

#[inline]
pub(crate) fn pad_const_inner<'a, S, T>(
    data: &ArrayBase<S, Ix2>,
    mut buf: ArrayViewMut2<'a, T>,
    padding: &ExplicitPadding<N>,
) -> ArrayViewMut2<'a, T>
where
    S: ndarray::Data<Elem = T>,
    T: Copy + NumAssign + std::fmt::Debug,
{
    buf.slice_mut(s!(
        padding.pad[0][0]..data.shape()[0] + padding.pad[0][0],
        padding.pad[1][0]..data.shape()[1] + padding.pad[1][0],
    ))
    .assign(data);

    buf
}

fn pad_inner_row<'a, T>(
    mut pad_input: ArrayViewMut2<'a, T>,
    input_h: usize,
    input_w: usize,
    padding: &[[usize; 2]; N],
    row_border_type: &[BorderType<T>; 2],
) -> Option<ArrayViewMut2<'a, T>>
where
    T: Copy + NumAssign + std::fmt::Debug,
{
    let [padding_h, padding_w] = *padding;

    // if let ExplictPadding([row_border_type, _]) = border_type {
    match row_border_type[0] {
        BorderType::Const(num) => {
            // left padding
            pad_input
                .slice_mut(s!(padding_h[0]..input_h + padding_h[0], 0..padding_w[0]))
                .assign(&Array2::from_elem((input_h, padding_w[0]), num));
            // right padding
            pad_input
                .slice_mut(s!(
                    padding_h[0]..input_h + padding_h[0],
                    input_w + padding_w[0]..
                ))
                .assign(&Array2::from_elem((input_h, padding_w[1]), num));
        }
        BorderType::Reflect => {
            if padding_w[0] > input_w - 1 || padding_w[1] > input_h - 1 {
                return None;
            }
            for i in padding_h[0]..padding_h[0] + input_h {
                // left padding
                for j in 1..=padding_w[0] {
                    pad_input[[i, padding_w[0] - j]] = pad_input[[i, padding_w[0] + j]];
                }
                // right padding
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
        BorderType::Circular => unsafe {
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
        _ => return None,
    }

    pad_input.into()
}

fn pad_inner_col<'a, T>(
    mut pad_input: ArrayViewMut2<'a, T>,
    input_h: usize,
    input_w: usize,
    padding: &[[usize; 2]; N],
    col_border_type: &[BorderType<T>; 2],
) -> Option<ArrayViewMut2<'a, T>>
where
    T: Copy + NumAssign + std::fmt::Debug,
{
    let [padding_h, padding_w] = *padding;

    // if let PaddingMode::Custom([_, col_border_type]) = border_type {
    match col_border_type[0] {
        BorderType::Const(num) => {
            // top padding
            pad_input
                .slice_mut(s!(..padding_h[0], ..))
                .assign(&Array2::from_elem(
                    (padding_h[0], input_w + padding_w[0] + padding_w[1]),
                    num,
                ));
            // bottom padding
            pad_input
                .slice_mut(s!(padding_h[0] + input_h.., ..))
                .assign(&Array2::from_elem(
                    (padding_h[1], input_w + padding_w[0] + padding_w[1]),
                    num,
                ));
        }
        BorderType::Reflect => unsafe {
            if padding_h[0] > input_h - 1 || padding_h[1] > input_h - 1 {
                return None;
            }
            // top padding
            for i in 1..=padding_h[0] {
                let row_pad = pad_input.row(padding_h[0] + i);

                (&pad_input.row(padding_h[0] - i) as *const ArrayView1<T> as *mut ArrayViewMut1<T>)
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
        BorderType::Circular => unsafe {
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
        _ => return None,
    }

    pad_input.into()
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn padding_mode_unfold() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        dbg!(pad(
            &input_pixels,
            &PaddingSize::Custom([1, 2], [1, 1])
                .unfold(&std::array::from_fn(|i| kernel.shape()[i])),
            &PaddingMode::Const(7),
        ));

        dbg!(pad(
            &input_pixels,
            &PaddingSize::Custom([4, 4], [1, 1])
                .unfold(&std::array::from_fn(|i| kernel.shape()[i])),
            &PaddingMode::Reflect,
        ));
    }
}
