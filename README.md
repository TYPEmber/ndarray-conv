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
- [x] explict error type
- [x] bench with similar libs
- [x] support `Complex<T>`
- [ ] conv with GPU acceleration for N dimension `Array`/`ArrayView` via `wgpu`

## Examples

```rust
use ndarray_conv::*;

x_nd.conv(
    &k_n,
    ConvMode::Full,
    PaddingMode::Circular,
);

// for cross-correlation
x_nd.conv(
    k_n.no_reverse(),
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
// Example for thin wrapper
use ndarray::{
    array, Array, ArrayView, Dim, IntoDimension, Ix, RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use ndarray_conv::*;

pub fn fftconvolve<'a, T, const N: usize>(
    in1: impl Into<ArrayView<'a, T, Dim<[Ix; N]>>>,
    in2: impl Into<ArrayView<'a, T, Dim<[Ix; N]>>>,
) -> Array<T, Dim<[Ix; N]>>
where
    T: num::traits::NumAssign + rustfft::FftNum,
    Dim<[Ix; N]>: RemoveAxis,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
        SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
{
    in1.into()
        .conv_fft(&in2.into(), ConvMode::Full, PaddingMode::Zeros)
        .unwrap()
}

fn test() {
    let o0 = fftconvolve(&[1., 2.], &array![1., 3., 7.]);
    let o1 = fftconvolve(&vec![1., 2.], &[1., 3., 7.]);
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

fft_3d                  time:   [4.6476 ms 4.6651 ms 4.6826 ms]
fft_with_processor_3d   time:   [4.6393 ms 4.6575 ms 4.6754 ms]
torch_3d                time:   [160.73 ms 161.12 ms 161.56 ms]
fftconvolve_3d          time:   [11.991 ms 12.009 ms 12.031 ms]
```

## Versions
- 0.5.3 - Buf fix & Dependecy update.
- 0.5.2 - Doc update.
- 0.5.1 - Add support for Complex<T>. Complete unit tests. Improve performance.
- 0.5.0 - **[breaking change]** Add `ReverseKernel` trait for cross-correlation, make `conv` & `conv_fft` calculating mathematical convolution. 
- 0.4.2 - Remove `Debug` trait on `T`.
- 0.4.1 - Doc update.
- 0.4.0 - Dependency update: update ndarray from 0.15 to 0.16.
- 0.3.4 - Bug fix: fix unsafe type cast in circular padding.
- 0.3.3 - Bug fix: correct conv_fft's output shape.
- 0.3.2 - Improve performance, by modifying `good_fft_size` and `transpose`.
- 0.3.1 - Impl basic error type. Fix some bugs.
- 0.3.0 - update to N-Dimension convolution.
- 0.2.0 - finished `conv_2d` & `conv_2d_fft`.

## Frequently Asked Questions (FAQ)

This FAQ addresses common questions about the `ndarray-conv` crate, a Rust library for N-dimensional convolutions using the `ndarray` ecosystem.

### 1. What is `ndarray-conv`?

`ndarray-conv` is a Rust crate that provides N-dimensional convolution operations for the `ndarray` crate. It offers both standard and FFT-accelerated convolutions, giving you efficient tools for image processing, signal processing, and other applications that rely on convolutions.

### 2. What are the main features of `ndarray-conv`?

*   **N-Dimensional Convolutions:** Supports convolutions on arrays with any number of dimensions.
*   **Standard and FFT-Accelerated:** Offers both `conv` (standard) and `conv_fft` (FFT-based) methods.
*   **Flexible Convolution Modes:** `ConvMode` (Full, Same, Valid, Custom, Explicit) to control output size.
*   **Various Padding Modes:** `PaddingMode` (Zeros, Const, Reflect, Replicate, Circular, Custom, Explicit) to handle boundary conditions.
*   **Strides and Dilation:** Supports strided convolutions and dilated kernels using the `with_dilation()` method.
*   **Performance Optimization:** Uses FFTs for larger kernels and optimized low-level operations for efficiency. The `conv_fft_with_processor` allows to reuse an `FftProcessor` for improved performance on repeated calls.
*   **Integration with `ndarray`:** Seamlessly works with `ndarray` `Array` and `ArrayView` types.

### 3. When should I use `conv_fft` vs. `conv`?

*   **`conv_fft` (FFT-accelerated):** Generally faster for larger kernels (e.g., larger than 11x11) because the computational complexity of FFTs grows more slowly than direct convolution as the kernel size increases.
*   **`conv` (Standard):** Might be faster for very small kernels (e.g., 3x3, 5x5) due to the overhead associated with FFT calculations.

It's a good idea to benchmark both methods with your specific kernel sizes and data dimensions to determine the best choice.

### 4. How do I choose the right `ConvMode`?

*   **`Full`:** The output contains all positions where the kernel and input overlap at least partially. This results in the largest output size.
*   **`Same`:** The output has the same size as the input. This is achieved by padding the input appropriately.
*   **`Valid`:** The output contains only positions where the kernel and input fully overlap. This results in the smallest output size.
*   **`Custom`:** You specify the padding for all the dimensions and strides.
*   **`Explicit`:** You specify the explicit padding for each side of each dimension, and the strides.

The best choice depends on the desired output size and how you want to handle boundary conditions.

### 5. How do I handle border effects with `PaddingMode`?

`PaddingMode` determines how the input is padded before the convolution.

*   **`Zeros`:** Pads with zeros.
*   **`Const(value)`:** Pads with a constant value.
*   **`Reflect`:** Reflects the input at the borders.
*   **`Replicate`:** Replicates the edge values.
*   **`Circular`:** Treats the input as a circular buffer, wrapping around at the borders.
*   **`Custom`:** You provide an array of `BorderType` enums, one for each dimension, to specify different padding behavior for each dimension.
*   **`Explicit`:** You provide an array with arrays of `BorderType` enums, one for each side of each dimension, to specify different padding behavior for each dimension.

Choose the `PaddingMode` that best suits your application's requirements for handling edges.

### 6. What is dilation, and how do I use it?

Dilation expands the *receptive field* of a kernel without increasing the number of its parameters. It does this by inserting spaces (usually zeros) between the original kernel elements. A dilation factor of `d` means that `d-1` zeros are inserted between each kernel element.

*   Use the `with_dilation()` method on an `ndarray` `Array` or `ArrayView` representing your kernel to create a dilated kernel.
*   Pass the dilated kernel to the `conv` or `conv_fft` methods.

**Example:**

```rust
let kernel = ndarray::array![[1, 2, 3], [4, 5, 6]];
let dilated_kernel = kernel.with_dilation(2); // Dilate by a factor of 2 in both dimensions
// dilated_kernel will effectively be: [[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]
```

**Why Use Dilation?**

*   **Increased Receptive Field:** Captures information from a wider area of the input without increasing the parameter count.
*   **Computational Efficiency:** More efficient than using very large standard kernels to achieve the same receptive field.
*   **Multi-Scale Feature Extraction:** Enables extracting features at different scales by using varying dilation rates.

**Applications:**

*   Semantic segmentation
*   Object detection
*   Image generation
*   Audio processing
*   Time-series analysis

### 7. How can I improve the performance of repeated convolutions?

*   **Use `conv_fft_with_processor`:** If you're performing multiple FFT-based convolutions, create an `FftProcessor` and reuse it with the `conv_fft_with_processor` method. This avoids recomputing FFT plans and reallocating scratch buffers.
*   **Convert to `f32` or `f64`:** For FFT convolutions, ensure your input and kernel data are `f32` (for `Rfft32`) or `f64` (for `Rfft64`). This avoids unnecessary type conversions.

### 8. How do I install `ndarray-conv`?

Add the following to your `Cargo.toml` file:

```toml
ndarray-conv = "0.3.3"  # Use the latest version
```

### 9. Are there any limitations to be aware of?

*   **FFT Overhead:** For very small kernels, FFT-based convolutions might be slower than standard convolutions due to the overhead of FFT calculations.
*   **Memory Usage:** FFT operations might require additional memory for intermediate buffers.
*   **`conv_fft` requires floating point:** The input and kernel must be floating point types (`f32` or `f64`) for FFT-based convolutions.

### 10. How do I convert integer arrays to floating-point for use with `conv_fft`?

Use the `.map(|&x| x as f32)` or `.map(|&x| x as f64)` methods to convert an integer `ndarray` to `f32` or `f64`, respectively.

**Example:**

```rust
let int_array = ndarray::Array::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
let float_array = int_array.map(|&x| x as f32);
```

### 11. Where can I find examples and documentation?

*   **README:** The project's README file on GitHub contains basic examples and usage instructions.
*   **Rust Docs:** Once published to crates.io, you can find detailed API documentation on docs.rs.
*   **Test Cases:** The `tests` modules within the source code provide further examples of how to use the library.

### 12. How does `ndarray-conv` compare to other convolution libraries?

The `ndarray-conv` project includes benchmarks comparing its performance to libraries like `tch` (LibTorch/PyTorch), `ndarray-vision`, and `fftconvolve`. `ndarray-conv` is generally competitive and often outperforms these other libraries, especially when using `conv_fft_with_processor` for repeated convolutions.
