use ndarray_conv::conv_2d::fft::Conv2DFftExt;

fn main() {
    use ndarray::prelude::*;
    use ndarray_conv::conv_2d::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::time::Instant;

    let mut small_duration = 0u128;
    let test_cycles_small = 1;
    // small input images
    for _ in 0..test_cycles_small {
        let x = Array::random((2000, 5000), Uniform::new(0f32, 1.));
        let k = Array::random((21, 41), Uniform::new(0f32, 1.));

        let now = Instant::now();
        // x.conv_2d(&k);
        // naive_conv::conv_2d(&x, &k);
        x.conv_2d_fft(&k);
        // ndarray_conv::conv_2d::fft::conv_2d::<f64, ndarray::OwnedRepr<f64>, ndarray::OwnedRepr<f64>>(
        //     &x, &k,
        // );
        // ndarray_conv::conv_2d::ndrustfft::conv_2d::<f64, ndarray::OwnedRepr<f64>, ndarray::OwnedRepr<f64>>(
        //     &x, &k,
        // );
        small_duration += now.elapsed().as_nanos();
    }

    println!(
        "Time for small arrays, {} iterations: {} milliseconds",
        test_cycles_small,
        small_duration / 1_000_000
    );
}
