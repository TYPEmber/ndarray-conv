use super::Processor as ProcessorTrait;
use super::*;

pub struct Processor<T: FftNum> {
    cp: rustfft::FftPlanner<T>,
    _phantom: PhantomData<Complex<T>>,
}

impl<T: FftNum> Default for Processor<T> {
    fn default() -> Self {
        Self {
            cp: rustfft::FftPlanner::new(),
            _phantom: Default::default(),
        }
    }
}

impl<T: FftNum> Processor<T> {
    pub fn forward_internal<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        mut scratch: Option<&mut Vec<Complex<T>>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // Ensure we always run FFTs along the last axis (which is contiguous),
        // then permute the array so the next axis becomes the last.
        let mut output = input.to_owned();
        let mut output_shape: [usize; N] = std::array::from_fn(|i| output.raw_dim()[i]);

        // axes permutation helper: initial axes rotated right by 1
        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_right(1);

        // perform FFT on last axis, then permute and repeat for remaining axes
        for step in 0..N {
            let fft = self.cp.plan_fft_forward(output_shape[N - 1]);
            for mut lane in output.lanes_mut(ndarray::Axis(N - 1)) {
                match scratch.as_mut() {
                    Some(s) => fft.process_with_scratch(lane.as_slice_mut().unwrap(), s),
                    None => fft.process(lane.as_slice_mut().unwrap()),
                }
            }

            if step < N - 1 {
                output_shape.rotate_right(1);

                let mut buffer = Array::uninit(output_shape.into_dimension());
                buffer.zip_mut_with(&output.permuted_axes(axes), |transpose, &origin| {
                    transpose.write(origin);
                });
                output = unsafe { buffer.assume_init() };
            }
        }

        output
    }

    pub fn backward_internal<const N: usize>(
        &mut self,
        mut input: Array<Complex<T>, Dim<[Ix; N]>>,
        mut scratch: Option<&mut Vec<Complex<T>>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        // Reverse of forward: perform inverse FFTs along last axis and permute left
        let mut raw_dim: [usize; N] = std::array::from_fn(|i| input.raw_dim()[i]);

        let mut axes: [usize; N] = std::array::from_fn(|i| i);
        axes.rotate_left(1);

        // process N-1 axes with permutation between steps
        for _ in 0..N - 1 {
            let cp = self.cp.plan_fft_inverse(raw_dim[N - 1]);
            for mut lane in input.lanes_mut(ndarray::Axis(N - 1)) {
                match scratch.as_mut() {
                    Some(s) => cp.process_with_scratch(lane.as_slice_mut().unwrap(), s),
                    None => cp.process(lane.as_slice_mut().unwrap()),
                }
            }

            raw_dim.rotate_left(1);

            let mut buffer = Array::uninit(raw_dim.into_dimension());
            buffer.zip_mut_with(&input.permuted_axes(axes), |transpose, &origin| {
                transpose.write(origin);
            });
            input = unsafe { buffer.assume_init() };
        }

        // final axis
        let cp = self.cp.plan_fft_inverse(raw_dim[N - 1]);
        for mut lane in input.lanes_mut(ndarray::Axis(N - 1)) {
            match scratch.as_mut() {
                Some(s) => cp.process_with_scratch(lane.as_slice_mut().unwrap(), s),
                None => cp.process(lane.as_slice_mut().unwrap()),
            }
        }

        let len = T::from_usize(input.len()).unwrap();
        input.mapv(|x| x / Complex::new(len, T::zero()))
    }
}

impl<T: FftNum> ProcessorTrait<T, Complex<T>> for Processor<T> {
    #[allow(clippy::uninit_vec)]
    fn get_scratch<const N: usize>(&mut self, input_dim: [usize; N]) -> Vec<Complex<T>> {
        let max_len = input_dim
            .iter()
            .copied()
            .map(|dim| self.cp.plan_fft_forward(dim).get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let mut scratch = Vec::with_capacity(max_len);
        unsafe { scratch.set_len(max_len) };
        scratch
    }

    fn forward<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.forward_internal(input, None)
    }

    fn backward<const N: usize>(
        &mut self,
        input: Array<Complex<T>, Dim<[Ix; N]>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.backward_internal(input, None)
    }

    fn forward_with_scratch<S: DataMut<Elem = Complex<T>>, const N: usize>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.forward_internal(input, Some(scratch))
    }

    fn backward_with_scratch<const N: usize>(
        &mut self,
        input: Array<Complex<T>, Dim<[Ix; N]>>,
        scratch: &mut Vec<Complex<T>>,
    ) -> Array<Complex<T>, Dim<[Ix; N]>>
    where
        Dim<[Ix; N]>: RemoveAxis,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    {
        self.backward_internal(input, Some(scratch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_forward_backward_1d() {
        let mut proc = Processor::<f32>::default();
        let arr = array![
            Complex::new(1.0f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ];
        let freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(freq);
        for (a, b) in arr.iter().zip(recon.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn test_forward_backward_2d() {
        let mut proc = Processor::<f32>::default();
        let arr = array![
            [Complex::new(1.0f32, 0.0), Complex::new(2.0, 0.0)],
            [Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]
        ];
        let freq = proc.forward(&mut arr.clone());
        let recon = proc.backward(freq);
        for (a, b) in arr.iter().zip(recon.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn test_scratch_methods() {
        let mut proc = Processor::<f32>::default();
        let input_dim = [4, 2];
        let mut arr = Array2::<Complex<f32>>::zeros((4, 2));
        let mut scratch = proc.get_scratch(input_dim);
        let freq = proc.forward_with_scratch(&mut arr, &mut scratch);
        let recon = proc.backward_with_scratch(freq, &mut scratch);
        for v in recon.iter() {
            assert!(v.re.abs() < 1e-4 && v.im.abs() < 1e-4);
        }
    }
}
