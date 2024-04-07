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
- [ ] handle input size error
- [ ] explict error type
- [ ] bench with similar libs

## Example

```rust
use ndarray_conv::*;

x_nd.conv(
    &k_n,
    PaddingSize::Full,
    PaddingMode::Circular,
);

x_1d.conv_fft(
    &k_1d,
    ConvMode::Same,
    PaddingMode::Explicit([[BorderType::Replicate, BorderType::Reflect]]),
);

x_2d.conv_fft(
    k_2d.with_dilation(2),
    PaddingSize::Same,
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
