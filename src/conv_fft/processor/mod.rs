//! Provides FFT processor implementations for convolution operations.
//!
//! This module contains traits and implementations for performing forward and backward FFT transforms
//! on real and complex-valued arrays. These processors are used internally by the FFT-accelerated
//! convolution methods.

use std::marker::PhantomData;

use ndarray::{Array, ArrayBase, DataMut, Dim, IntoDimension, Ix, RemoveAxis};
use num::Complex;
use rustfft::FftNum;

pub mod complex;
pub mod real;

/// Marker trait for numeric types that can be used with ConvFftNum.
///
/// This trait is implemented for both integer and floating-point types that implement `FftNum`.
///
/// # Important Note
///
/// While this trait is implemented for integer types (i8, i16, i32, i64, i128, isize),
/// **integer FFT operations have known accuracy issues** and should NOT be used in production.
/// Only lengths of 2 or 4 work correctly for 1D arrays; other lengths produce incorrect results.
///
/// **Always use f32 or f64 for FFT operations.**
pub trait ConvFftNum: FftNum {}

macro_rules! impl_conv_fft_num {
    ($($t:ty),*) => {
        $(impl ConvFftNum for $t {})*
    };
}

impl_conv_fft_num!(i8, i16, i32, i64, i128, isize, f32, f64);

/// Returns a processor instance for the given input element type.
///
/// This function is a convenience wrapper around `GetProcessor::get_processor()`.
///
/// # Type Parameters
///
/// * `T`: The FFT numeric type (typically f32 or f64)
/// * `InElem`: The input element type (`T` for real, `Complex<T>` for complex)
///
/// # Example
///
/// ```rust
/// use ndarray_conv::GetProcessor;
///
/// // Get a processor for f32 real values
/// let mut proc = GetProcessor::<f32, f32>::get_processor();
///
/// // Get a processor for Complex<f32> values
/// use num::Complex;
/// let mut proc_complex = GetProcessor::<f32, Complex<f32>>::get_processor();
/// ```
pub fn get<T: FftNum, InElem: GetProcessor<T, InElem>>() -> impl Processor<T, InElem> {
    InElem::get_processor()
}

/// Trait for FFT processors that can perform forward and backward transforms.
///
/// This trait defines the interface for performing FFT operations on N-dimensional arrays.
/// Implementations exist for both real-valued and complex-valued inputs.
pub trait Processor<T: FftNum, InElem: GetProcessor<T, InElem>> {
    /// Performs a forward FFT transform.
    ///
    /// Converts the input array from the spatial/time domain to the frequency domain.
    ///
    /// # Arguments
    ///
    /// * `input`: A mutable reference to the input array
    ///
    /// # Returns
    ///
    /// An array of complex values representing the frequency domain.
    fn forward<S: DataMut<Elem = InElem>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;

    /// Performs a backward (inverse) FFT transform.
    ///
    /// Converts the input array from the frequency domain back to the spatial/time domain.
    ///
    /// # Arguments
    ///
    /// * `input`: A mutable reference to the frequency domain array
    ///
    /// # Returns
    ///
    /// An array in the spatial/time domain with the same element type as the original input.
    fn backward<const N: usize>(
        &mut self,
        input: &mut Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<InElem, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;
}

/// Trait for types that can provide a processor instance.
///
/// This trait is implemented for real and complex numeric types, allowing them to
/// automatically select the appropriate FFT processor implementation.
pub trait GetProcessor<T: FftNum, InElem>
where
    InElem: GetProcessor<T, InElem>,
{
    /// Returns a processor instance appropriate for this type.
    fn get_processor() -> impl Processor<T, InElem>;
}

impl<T: ConvFftNum> GetProcessor<T, T> for T {
    fn get_processor() -> impl Processor<T, T> {
        real::Processor::<T>::default()
    }
}

impl<T: FftNum> GetProcessor<T, Complex<T>> for Complex<T> {
    fn get_processor() -> impl Processor<T, Complex<T>> {
        complex::Processor::<T>::default()
    }
}
