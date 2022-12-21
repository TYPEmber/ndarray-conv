# ndarray-conv

ndarray-conv is a crate that provides a fast convolutions library in pure Rust.

Inspired by

ndarray-vision (https://github.com/rust-cv/ndarray-vision)

convolutions-rs (https://github.com/Conzel/convolutions-rs#readme)

pocketfft (https://github.com/mreineck/pocketfft)

ndarray-conv is still under heavily developing, the first stage aims to provide a fast conv_2d func for ndarray::Array2<T>.

## First Stage

- [x] basic conv_2d
- [x] use rayon to accelerate big matrix's conv_2d computation
- [ ] use fft to accelerate big kernel's conv_2d computation

## Roughly Bench

**without rayon**

2x-4x faster than ndarray-vision and 4x-10x faster than convolutions-rs.
2x-4x slower than opencv with small kernel (size < 11)

**with rayon**

2x faster than opencv with small kernel (size < 11)

## Example

```rust
    use ndarray_conv::conv_2d::*;
    x.conv_2d(&k);
```

```rust
fn main() {
    use ndarray_conv::conv_2d::*;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::time::Instant;

    let mut small_duration = 0u128;
    let test_cycles_small = 1;
    // small input images
    for _ in 0..test_cycles_small {
        let x = Array::random((2000, 4000), Uniform::new(0., 1.));
        let k = Array::random((9, 9), Uniform::new(0., 1.));

        let now = Instant::now();
        x.conv_2d(&k);
        small_duration += now.elapsed().as_nanos();
    }

    println!(
        "Time for small arrays, {} iterations: {} milliseconds",
        test_cycles_small,
        small_duration / 1_000_000
    );
}
```
