use std::marker::PhantomData;

use ndarray::{Array, ArrayBase, DataMut, Dim, IntoDimension, Ix, RemoveAxis};
use num::{traits::real::Real, Complex};
use rustfft::{num_complex, num_traits, FftNum};

pub mod complex;
pub mod real;

pub trait ConvFftNum: FftNum {}

macro_rules! impl_conv_fft_num {
    ($($t:ty),*) => {
        $(impl ConvFftNum for $t {})*
    };
}

impl_conv_fft_num!(i8, i16, i32, isize, f32, f64);

pub trait Processor<T: FftNum, InElem> {
    fn get_scratch<const N: usize>(&mut self, input_dim: [usize; N]) -> Vec<Complex<T>>;

    fn forward<S: DataMut<Elem = InElem>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;

    fn backward<const N: usize>(
        &mut self,
        input: Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<InElem, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;

    fn forward_with_scratch<S: DataMut<Elem = InElem>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;

    fn backward_with_scratch<const N: usize>(
        &mut self,
        input: Array<Complex<T>, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<InElem, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>;
}

// impl<T: FftNum> Processor for ComplexProcessor<T> {}

// pub trait GetProcessor {
//     fn get_processor() -> impl Processor;
// }

// impl<T: ConvFftNum> GetProcessor for T {
//     fn get_processor() -> impl Processor {
//         RealProcessor::<T>::default()
//     }
// }

// impl<CT> GetProcessor for CT
// where
//     CT: Into<Complex<f64>>,
// {
//     fn get_processor() -> impl Processor {
//         let a:Complex<f64> = 1f64.into();
//         ComplexProcessor::<f64>::default()
//     }
// }

// impl GetProcessor for Complex<f64>
// {
//     fn get_processor() -> impl Processor {
//         let a:Complex<f64> = 1f64.into();
//         ComplexProcessor::<f64>::default()
//     }
// }

pub trait Marker {}

// impl Marker for Complex<f64> {}

// impl GetProcessor for Complex<f64> {
//     fn get_processor() -> impl Processor {
//         ComplexProcessor::<f64>::default()
//     }
// }
