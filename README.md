# ndarray-conv

ndarray-conv is a crate that provides a N-Dimension convolutions (with FFT acceleration) library in pure Rust.

Inspired by

ndarray-vision (https://github.com/rust-cv/ndarray-vision)

convolutions-rs (https://github.com/Conzel/convolutions-rs#readme)

pocketfft (https://github.com/mreineck/pocketfft)

## Roadmap

- [x] basic conv for N dimension `Array`/`ArrayView`
- [x] conv with FFT acceleration for N dimension `Array`/`ArrayView`
- [x] impl `ConvMode` and `PaddingMode`
  - [x] `ConvMode`: Full Same Valid Custom Explicit
  - [x] `PaddingMode`: Zeros Const Reflect Replicate Circular Custom Explicit
- [x] conv with strides
- [x] kernel with dilation
- [x] handle input size error
- [ ] explict error type
- [ ] bench with similar libs

## Examples

```rust
use ndarray_conv::*;

x_nd.conv(
    &k_n,
    ConvMode::Full,
    PaddingMode::Circular,
);

x_1d.view().conv_fft(
    &k_1d,
    ConvMode::Same,
    PaddingMode::Explicit([[BorderType::Replicate, BorderType::Reflect]]),
);

x_2d.conv_fft(
    k_2d.with_dilation(2),
    ConvMode::Same,
    PaddingMode::Custom([BorderType::Reflect, BorderType::Circular]),
);

// avoid loss of accuracy for fft ver
// convert Integer to Float before caculate.
x_3d.map(|&x| x as f32)
    .conv_fft(
        &kernel.map(|&x| x as f32),
        ConvMode::Same,
        PaddingMode::Zeros,
    )
    .unwrap()
    .map(|x| x.round() as i32);
```

```rust
fn main() {
    use ndarray_conv::*;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::time::Instant;

    let mut small_duration = 0u128;
    let test_cycles_small = 1;
    // small input data
    for _ in 0..test_cycles_small {
        let x = Array::random((2000, 4000), Uniform::new(0., 1.));
        let k = Array::random((9, 9), Uniform::new(0., 1.));

        let now = Instant::now();
        // or use x.conv_fft() for large input data
        x.conv(
            &k,
            ConvMode::Same,
            PaddingMode::Custom([BorderType::Reflect, BorderType::Circular]),
        );
        small_duration += now.elapsed().as_nanos();
    }

    println!(
        "Time for small arrays, {} iterations: {} milliseconds",
        test_cycles_small,
        small_duration / 1_000_000
    );
}
```

## Benchmark

```rust
let x = Array::random(5000, Uniform::new(0f32, 1.));
let k = Array::random(31, Uniform::new(0f32, 1.));

fft_1d                  time:   [76.621 µs 76.649 µs 76.681 µs]
fft_with_processor_1d   time:   [34.563 µs 34.790 µs 35.125 µs]
torch_1d                time:   [45.542 µs 45.658 µs 45.775 µs]
fftconvolve_1d          time:   [161.52 µs 162.28 µs 163.05 µs]

---------------------------------------------------------------

let x = Array::random((200, 5000), Uniform::new(0f32, 1.));
let k = Array::random((11, 31), Uniform::new(0f32, 1.));

fft_2d                  time:   [16.022 ms 16.046 ms 16.071 ms]
fft_with_processor_2d   time:   [15.949 ms 15.977 ms 16.010 ms]
torch_2d                time:   [109.76 ms 111.62 ms 113.79 ms]
ndarray_vision_2d       time:   [429.47 ms 429.64 ms 429.82 ms]
fftconvolve_2d          time:   [56.273 ms 56.342 ms 56.420 ms]

---------------------------------------------------------------

let x = Array::random((10, 100, 200), Uniform::new(0f32, 1.));
let k = Array::random((5, 11, 31), Uniform::new(0f32, 1.));

fft_3d                  time:   [5.3049 ms 5.3498 ms 5.3957 ms]
fft_with_processor_3d   time:   [5.2981 ms 5.3345 ms 5.3696 ms]
torch_3d                time:   [147.20 ms 151.97 ms 158.54 ms]
fftconvolve_3d          time:   [11.991 ms 12.009 ms 12.031 ms]
```

## Versions
- 0.3.3 - Bug fix: correct conv_fft's output shape.
- 0.3.2 - Improve performance, by modifying `good_fft_size` and `transpose`.
- 0.3.1 - Impl basic error type. Fix some bugs.
- 0.3.0 - update to N-Dimension convolution.
- 0.2.0 - finished `conv_2d` & `conv_2d_fft`.