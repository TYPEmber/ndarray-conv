use std::fmt::Display;

use ndarray::prelude::*;
use num::traits::NumAssign;

fn conv_2d<T, S: ndarray::Data<Elem = T>, SM: ndarray::Data<Elem = T>>(
    data: &ArrayBase<S, Ix2>,
    kernel: &ArrayBase<S, Ix2>,
    ret: &mut ArrayBase<SM, Ix2>,
) -> Option<()>
where
    T: Copy + Clone + NumAssign + std::fmt::Debug + Display + Send + Sync,
    f64: std::convert::From<T>,
{
    let data_spec = fft_2d(data);
    let kernel_spec = fft_2d(kernel);
    None
}

fn fft_2d<T, S: ndarray::Data<Elem = T>>(
    arr: &ArrayBase<S, Ix2>,
) -> Array2<rustfft::num_complex::Complex64>
where
    T: Copy + Clone + NumAssign + std::fmt::Debug + Display + Send + Sync,
    f64: std::convert::From<T>,
{
    let mut planner = rustfft::FftPlanner::new();
    let mut data_complex = arr.mapv(|x| rustfft::num_complex::Complex64 {
        re: x.into(),
        im: 0.0,
    });
    let fft_row = planner.plan_fft_forward(data_complex.shape()[1]);
    let fft_col = planner.plan_fft_forward(data_complex.shape()[0]);

    for mut row in data_complex.rows_mut() {
        fft_row.process(row.as_slice_mut().unwrap());
    }

    for mut col in data_complex
        .t()
        .as_standard_layout()
        .reversed_axes()
        .columns_mut()
    {
        fft_col.process(col.as_slice_memory_order_mut().unwrap());
    }

    data_complex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let mut ret = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];

        for a in input_pixels.axis_iter(Axis(0)) {
            dbg!(a);
        }

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];

        conv_2d(&input_pixels, &kernel, &mut ret);
    }
}
