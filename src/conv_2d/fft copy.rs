use std::fmt::Display;

use ndarray::prelude::*;
use num::{traits::NumAssign, ToPrimitive};
use rustfft::num_traits::FromPrimitive;

pub fn conv_2d<T, S: ndarray::Data<Elem = T>, SM: ndarray::Data<Elem = T>>(
    data: &ArrayBase<S, Ix2>,
    kernel: &ArrayBase<S, Ix2>,
) -> Option<Array2<T>>
where
    T: Copy
        + Clone
        + NumAssign
        + std::fmt::Debug
        + Display
        + Send
        + Sync
        + FromPrimitive
        + ToPrimitive,

    f64: std::convert::From<T>,
{
    // let data_spec = fft_2d(data);

    // let mut kernel_ext = Array2::zeros(data.dim());
    // kernel_ext
    //     .slice_mut(s![..kernel.shape()[0], ..kernel.shape()[1]])
    //     .assign(kernel);
    // let kernel_spec = fft_2d(&kernel_ext);

    // let mut mul_spec = data_spec * kernel_spec;

    // Some(ifft_2d(&mut mul_spec))

    use ndrustfft::{ndfft, ndifft, Complex, FftHandler};

    let mut data_ext = Array2::zeros((
        data.shape()[0] + kernel.shape()[0] - 1,
        data.shape()[1] + kernel.shape()[1] - 1,
    ));
    data_ext
        .slice_mut(s!(..data.shape()[0], ..data.shape()[1]))
        .assign(data);

    let (nx, ny) = data_ext.dim();
    let mut data_ext = data_ext.mapv(|x| rustfft::num_complex::Complex64 {
        re: x.into(),
        im: 0.0,
    });
    let mut buf = Array2::<Complex<f64>>::zeros((nx, ny));
    let mut data_spec = Array2::<Complex<f64>>::zeros((nx, ny));
    let mut fft_handler_x = FftHandler::<f64>::new(nx);
    let mut fft_handler_y = FftHandler::<f64>::new(ny);
    ndfft(
        &data_ext.view(),
        &mut buf.view_mut(),
        &mut fft_handler_x,
        0,
    );
    ndfft(
        &buf.view(),
        &mut data_spec.view_mut(),
        &mut fft_handler_y,
        1,
    );

    let mut kernel_ext = Array2::zeros(data_ext.dim());
    kernel_ext
        .slice_mut(s![..kernel.shape()[0], ..kernel.shape()[1]])
        .assign(kernel);
    let (nx, ny) = kernel_ext.dim();
    let mut kernel_ext = kernel_ext.mapv(|x| rustfft::num_complex::Complex64 {
        re: x.into(),
        im: 0.0,
    });
    let mut kernel_spec = Array2::<Complex<f64>>::zeros((nx, ny));
    // let mut fft_handler_x = FftHandler::<f64>::new(nx);
    // let mut fft_handler_y = FftHandler::<f64>::new(ny);
    ndfft(
        &kernel_ext.view(),
        &mut buf.view_mut(),
        &mut fft_handler_x,
        0,
    );
    ndfft(
        &buf.view(),
        &mut kernel_spec.view_mut(),
        &mut fft_handler_y,
        1,
    );

    let mut mul_spec = data_spec * kernel_spec;
    let mut ret_buf = Array2::zeros(data_ext.dim());
    let mut ret = Array2::zeros(data_ext.dim());
    ndifft(&mul_spec.view(), &mut ret_buf, &mut fft_handler_y, 1);
    ndifft(&ret_buf.view(), &mut ret, &mut fft_handler_x, 0);

    Some(ret.mapv(|x| T::from_f64(x.re.round()).unwrap()))
}

fn ifft_2d<T>(arr: &mut Array2<rustfft::num_complex::Complex64>) -> Array2<T>
where
    T: Copy + Clone + NumAssign + std::fmt::Debug + Display + Send + Sync + FromPrimitive,
    f64: std::convert::From<T>,
{
    let mut planner = rustfft::FftPlanner::new();

    let ifft_row = planner.plan_fft_inverse(arr.shape()[1]);
    let ifft_col = planner.plan_fft_inverse(arr.shape()[0]);

    for mut row in arr.rows_mut() {
        ifft_row.process(row.as_slice_mut().unwrap());
    }

    for mut col in arr.t().as_standard_layout().reversed_axes().columns_mut() {
        ifft_col.process(col.as_slice_memory_order_mut().unwrap());
    }

    arr.mapv(|x| T::from_f64(x.re / 5 as f64).unwrap())
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

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0]];

        dbg!(&conv_2d::<
            i32,
            ndarray::OwnedRepr<i32>,
            ndarray::OwnedRepr<i32>,
        >(&input_pixels, &kernel));
    }
}
