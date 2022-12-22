use ndarray::prelude::*;
use num::traits::*;
use std::fmt::{Debug, Display};
use transpose::transpose;
pub fn inverse<T>(
    arr: &mut Array2<rustfft::num_complex::Complex32>,
    r_planner: &mut realfft::RealFftPlanner<f32>,
    c_planner: &mut rustfft::FftPlanner<f32>,
) -> Array2<T>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + FromPrimitive,
{
    // return Array2::zeros((1,1));

    let ifft_row = r_planner.plan_fft_inverse((arr.shape()[0] - 1) * 2);
    let ifft_col = c_planner.plan_fft_inverse(arr.shape()[1]);

    ifft_col.process(arr.as_slice_mut().unwrap());

    // ndarray::Zip::from(arr.rows_mut()).par_for_each(|mut row| {
    //     ifft_col.process(row.as_slice_mut().unwrap());
    // });

    let mut output_t = Array2::zeros((arr.shape()[1], (arr.shape()[0] - 1) * 2));

    // for mut row in arr.rows_mut() {
    //     ifft_col.process(row.as_slice_mut().unwrap());
    // }

    {
        let mut input_t = Array2::zeros((arr.shape()[1], arr.shape()[0]));

        transpose(
            arr.as_slice().unwrap(),
            input_t.as_slice_mut().unwrap(),
            arr.shape()[1],
            arr.shape()[0],
        );

        // input_t
        //     .as_slice_mut()
        //     .unwrap()
        //     .chunks_exact_mut(arr.shape()[0])
        //     .zip(
        //         output_t
        //             .as_slice_mut()
        //             .unwrap()
        //             .chunks_exact_mut((arr.shape()[0] - 1) * 2 + 1),
        //     )
        //     .for_each(|(row, output)| {
        //         unsafe { row.get_unchecked_mut(0).im = 0.0 };
        //         // row.last_mut().unwrap().im = 0.0;
        //         ifft_row.process(row, output).unwrap();
        //     });

        ndarray::Zip::from(input_t.rows_mut())
            .and(output_t.rows_mut())
            .for_each(|mut row, mut output| {
                unsafe { row.uget_mut(0).im = 0.0 };
                unsafe { row.uget_mut(row.len() - 1).im = 0.0 };
                // row.last_mut().unwrap().im = 0.0;
                ifft_row
                    .process(row.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
                    .unwrap();
            });
    }

    // {
    //     ndarray::Zip::from(arr.columns_mut())
    //         .and(output_t.rows_mut())
    //         .for_each(|mut row, mut output| {
    //             let mut row = row.mapv(|x| x);
    //             unsafe { row.uget_mut(0).im = 0.0 };
    //             // row.last_mut().unwrap().im = 0.0;
    //             ifft_row
    //                 .process(row.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
    //                 .unwrap();
    //         });
    // }

    // output_t.mapv(|x| T::from_f32((x / output_t.len() as f32).round()).unwrap())
    output_t.mapv(|x| T::from_f32(x / output_t.len() as f32 + 0.5).unwrap())
}

pub fn forward<T, S: ndarray::Data<Elem = T>>(
    arr: &ArrayBase<S, Ix2>,
    r_planner: &mut realfft::RealFftPlanner<f32>,
    c_planner: &mut rustfft::FftPlanner<f32>,
) -> Array2<rustfft::num_complex::Complex32>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + AsPrimitive<f32>,
{
    let (nx, ny) = arr.dim();
    let mut input = arr.mapv(|x| x.as_());
    let mut output = Array2::zeros((nx, ny / 2 + 1));
    let fft_row = r_planner.plan_fft_forward(ny);
    let fft_col = c_planner.plan_fft_forward(nx);

    ndarray::Zip::from(input.rows_mut())
        .and(output.rows_mut())
        .for_each(|mut row, mut output| {
            fft_row
                .process(row.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
                .unwrap();
        });

    // input
    //     .as_slice_mut()
    //     .unwrap()
    //     .chunks_exact_mut(ny)
    //     .zip(output.as_slice_mut().unwrap().chunks_exact_mut(ny / 2 + 1))
    //     .for_each(|(row, output)| {
    //         fft_row.process(row, output).unwrap();
    //     });

    let output = output.into_raw_vec();
    let mut output_t = vec![rustfft::num_complex::Complex::default(); output.len()];
    transpose::transpose(&output, &mut output_t, ny / 2 + 1, nx);

    fft_col.process(&mut output_t);

    let mut output_t = Array2::from_shape_vec((ny / 2 + 1, nx), output_t).unwrap();

    // ndarray::Zip::from(output_t.rows_mut()).par_for_each(|mut row| {
    //     fft_col.process(row.as_slice_mut().unwrap());
    // });

    // for mut row in output_t.rows_mut() {
    //     fft_col.process(row.as_slice_mut().unwrap());
    // }

    output_t
}

pub fn inverse_f64<T>(
    arr: &mut Array2<rustfft::num_complex::Complex64>,
    r_planner: &mut realfft::RealFftPlanner<f64>,
    c_planner: &mut rustfft::FftPlanner<f64>,
) -> Array2<T>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync + FromPrimitive,
    f64: std::convert::From<T>,
{
    let ifft_row = r_planner.plan_fft_inverse((arr.shape()[0] - 1) * 2);
    let ifft_col = c_planner.plan_fft_inverse(arr.shape()[1]);

    ifft_col.process(arr.as_slice_mut().unwrap());

    let mut input_t = Array2::zeros((arr.shape()[1], arr.shape()[0]));
    let mut output_t = Array2::zeros((arr.shape()[1], (arr.shape()[0] - 1) * 2));

    transpose(
        arr.as_slice().unwrap(),
        input_t.as_slice_mut().unwrap(),
        arr.shape()[1],
        arr.shape()[0],
    );

    ndarray::Zip::from(input_t.rows_mut())
        .and(output_t.rows_mut())
        .for_each(|mut row, mut output| {
            row.first_mut().unwrap().im = 0.0;
            // row.last_mut().unwrap().im = 0.0;
            ifft_row
                .process(row.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
                .unwrap();
        });

    // output_t.mapv(|x| T::from_f64((x / output_t.len() as f64).round()).unwrap())
    output_t.mapv(|x| T::from_f64(x / output_t.len() as f64 + 0.5).unwrap())
}

pub fn forward_f64<T, S: ndarray::Data<Elem = T>>(
    arr: &ArrayBase<S, Ix2>,
    r_planner: &mut realfft::RealFftPlanner<f64>,
    c_planner: &mut rustfft::FftPlanner<f64>,
) -> Array2<rustfft::num_complex::Complex64>
where
    T: Copy + Clone + NumAssign + Debug + Display + Send + Sync,
    f64: std::convert::From<T>,
{
    let (nx, ny) = arr.dim();
    let mut input = arr.mapv(|x| x.into());
    let mut output = Array2::zeros((nx, ny / 2 + 1));
    let fft_row = r_planner.plan_fft_forward(ny);
    let fft_col = c_planner.plan_fft_forward(nx);

    ndarray::Zip::from(input.rows_mut())
        .and(output.rows_mut())
        .for_each(|mut row, mut output| {
            fft_row
                .process(row.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
                .unwrap();
        });

    let output = output.into_raw_vec();
    let mut output_t = vec![rustfft::num_complex::Complex::default(); output.len()];
    transpose::transpose(&output, &mut output_t, ny / 2 + 1, nx);

    fft_col.process(&mut output_t);

    Array2::from_shape_vec((ny / 2 + 1, nx), output_t).unwrap()
}
