use ndarray::prelude::*;
use num::traits::*;
use std::fmt::{Debug, Display};
pub fn inverse<T>(arr: &mut Array2<rustfft::num_complex::Complex<f32>>, planner: &mut rustfft::FftPlanner<f32>) -> Array2<T>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + FromPrimitive,
{
    // let mut planner = rustfft::FftPlanner::new();

    // let mut arr = arr.t().to_owned();

    let ifft_row = planner.plan_fft_inverse(arr.shape()[0]);
    let ifft_col = planner.plan_fft_inverse(arr.shape()[1]);

    ifft_col.process(arr.as_slice_mut().unwrap());

    // for mut col in arr.columns_mut() {
    //     ifft_col.process(col.as_slice_mut().unwrap());
    // }
    let mut output = vec![rustfft::num_complex::Complex::default(); arr.len()];
    transpose::transpose(
        arr.as_slice().unwrap(),
        &mut output,
        arr.shape()[1],
        arr.shape()[0],
    );

    // let mut arr = arr.as_standard_layout().to_owned();

    // for mut row in arr.rows_mut() {
    //     ifft_row.process(row.as_slice_mut().unwrap());
    // }

    ifft_row.process(&mut output);

    Array2::from_shape_vec(
        (arr.shape()[1], arr.shape()[0]),
        output
            .iter()
            .map(|x| T::from_f32((x.re / arr.len() as f32) + 0.5).unwrap())
            .collect(),
    )
    .unwrap()
}

pub fn forward<T, S: ndarray::Data<Elem = T>>(
    arr: &ArrayBase<S, Ix2>,
    planner: &mut rustfft::FftPlanner<f32>
) -> Array2<rustfft::num_complex::Complex<f32>>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + AsPrimitive<f32>,
{
    // let mut planner = rustfft::FftPlanner::new();
    let mut data_complex = arr.mapv(|x| rustfft::num_complex::Complex {
        re: x.as_(),
        im: 0.0,
    });
    let fft_row = planner.plan_fft_forward(arr.shape()[1]);
    let fft_col = planner.plan_fft_forward(arr.shape()[0]);

    // for mut row in data_complex.rows_mut() {
    //     fft_row.process(row.as_slice_mut().unwrap());
    // }

    fft_row.process(data_complex.as_slice_mut().unwrap());

    let mut data_complex_t = vec![rustfft::num_complex::Complex::default(); data_complex.len()];
    transpose::transpose(
        data_complex.as_slice().unwrap(),
        &mut data_complex_t,
        data_complex.shape()[1],
        data_complex.shape()[0],
    );

    // let mut data_complex = data_complex
    //     .t()
    //     .as_standard_layout()
    //     .reversed_axes()
    //     .to_owned();

    // for mut col in data_complex.columns_mut() {
    //     fft_col.process(col.as_slice_memory_order_mut().unwrap());
    // }

    // data_complex

    fft_col.process(&mut data_complex_t);

    Array2::from_shape_vec(
        (data_complex.shape()[1], data_complex.shape()[0]),
        data_complex_t,
    )
    .unwrap()
}




pub fn inverse_f64<T>(arr: &mut Array2<rustfft::num_complex::Complex64>, planner: &mut rustfft::FftPlanner<f64>) -> Array2<T>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + FromPrimitive,
    f64: std::convert::From<T>,
{
    // let mut planner = rustfft::FftPlanner::new();

    // let mut arr = arr.t().to_owned();

    let ifft_row = planner.plan_fft_inverse(arr.shape()[0]);
    let ifft_col = planner.plan_fft_inverse(arr.shape()[1]);

    ifft_col.process(arr.as_slice_mut().unwrap());

    // for mut col in arr.columns_mut() {
    //     ifft_col.process(col.as_slice_mut().unwrap());
    // }
    let mut output = vec![rustfft::num_complex::Complex64::default(); arr.len()];
    transpose::transpose(
        arr.as_slice().unwrap(),
        &mut output,
        arr.shape()[1],
        arr.shape()[0],
    );

    // let mut arr = arr.as_standard_layout().to_owned();

    // for mut row in arr.rows_mut() {
    //     ifft_row.process(row.as_slice_mut().unwrap());
    // }

    ifft_row.process(&mut output);

    Array2::from_shape_vec(
        (arr.shape()[1], arr.shape()[0]),
        output
            .iter_mut()
            .map(|x| T::from_f64((x.re / arr.len() as f64).round()).unwrap())
            .collect(),
    )
    .unwrap()
}

pub fn forward_f64<T, S: ndarray::Data<Elem = T>>(
    arr: &ArrayBase<S, Ix2>,
    planner: &mut rustfft::FftPlanner<f64>
) -> Array2<rustfft::num_complex::Complex64>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync,
    f64: std::convert::From<T>,
{
    // let mut planner = rustfft::FftPlanner::new();
    let mut data_complex = arr.mapv(|x| rustfft::num_complex::Complex64 {
        re: x.into(),
        im: 0.0,
    });
    let fft_row = planner.plan_fft_forward(data_complex.shape()[1]);
    let fft_col = planner.plan_fft_forward(data_complex.shape()[0]);

    // for mut row in data_complex.rows_mut() {
    //     fft_row.process(row.as_slice_mut().unwrap());
    // }

    fft_row.process(data_complex.as_slice_mut().unwrap());

    let mut data_complex_t = vec![rustfft::num_complex::Complex64::default(); data_complex.len()];
    transpose::transpose(
        data_complex.as_slice().unwrap(),
        &mut data_complex_t,
        data_complex.shape()[1],
        data_complex.shape()[0],
    );

    // let mut data_complex = data_complex
    //     .t()
    //     .as_standard_layout()
    //     .reversed_axes()
    //     .to_owned();

    // for mut col in data_complex.columns_mut() {
    //     fft_col.process(col.as_slice_memory_order_mut().unwrap());
    // }

    // data_complex

    fft_col.process(&mut data_complex_t);

    Array2::from_shape_vec(
        (data_complex.shape()[1], data_complex.shape()[0]),
        data_complex_t,
    )
    .unwrap()
}
