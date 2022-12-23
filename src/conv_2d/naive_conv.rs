use std::vec;

use ndarray::prelude::*;

pub fn conv_2d<T: num::traits::NumAssign + Copy>(data: &Array2<T>, kernel: &Array2<T>) -> Array2<T> {
    // conv with same size output
    let (h, w) = (data.shape()[0], data.shape()[1]);
    let (kernel_h, kernel_w) = (kernel.shape()[0], kernel.shape()[1]);
    let (stride_h, stride_w) = (1, 1);
    let (pad_h, pad_w) = (
        (h * stride_h - h + kernel_h - 1) / 2,
        (w * stride_w - w + kernel_w - 1) / 2,
    );
    let (new_h, new_w) = (h + 2 * pad_h, w + 2 * pad_w);

    let mut pad_input = Array2::zeros((new_h, new_w));
    let mut sub_pad_input = pad_input.slice_mut(s!(pad_h..h + pad_h, pad_w..w + pad_w));
    sub_pad_input.assign(data);

    // let mut ret = Array2::zeros((h, w));
    // ndarray::Zip::from(&mut ret)
    //     .and(pad_input.windows((kernel_h, kernel_w)))
    //     .for_each(|r, w| {
    //         *r = (w.to_owned() * kernel).sum();
    //     });
    // ret

    let mut ret1 = Array2::zeros((1, h * w));

    let mut offsets = vec![];
    for i in 0..kernel_h {
        for j in 0..kernel_w {
            if kernel[(i, j)] == T::zero() {
                continue;
            }

            offsets.push((i * new_h + j, kernel[(i, j)]));
        }
    }
    // dbg!(input_offset);

    let mut ret_idx = 0;
    unsafe {
        for i in (0..=new_h - kernel_h).step_by(stride_h) {
            for j in (0..=new_w - kernel_w).step_by(stride_w) {
                let mut tmp_res = T::zero();
                let cur = pad_input.uget((i, j));
                for k in 0..offsets.len() {
                    let (tmp_offset, tmp_kernel) = offsets[k];

                    tmp_res += *(cur as *const T).add(tmp_offset) * tmp_kernel;
                }

                // for (tmp_offset, tmp_kernel) in offsets.iter() {
                //     tmp_res += *(cur as *const f32).add(*tmp_offset) * tmp_kernel;
                // }

                // let cur_input = pad_input.slice(s!(i..i + kernel_h, j..j + kernel_w));

                // *ret1.uget_mut((0, ret_idx)) = (cur_input.to_owned() * kernel).sum();
                // ret1[(0, ret_idx)] = (cur_input.to_owned() * kernel).sum();
                *ret1.uget_mut((0, ret_idx)) = tmp_res;
                ret_idx += 1;
            }
        }
    }
    ret1.into_shape((h, w)).unwrap()

    // let mut ret_idx = 0;
    // for i in (0..=new_h - kernel_h).step_by(stride_h) {
    //     for j in (0..=new_w - kernel_w).step_by(stride_w) {

    //         let cur_input = pad_input.slice(s!(i..i + kernel_h, j..j + kernel_w));

    //         // *ret1.uget_mut((0, ret_idx)) = (cur_input.to_owned() * kernel).sum();
    //         ret1[(0, ret_idx)] = (cur_input.to_owned() * kernel).sum();
    //         ret_idx += 1;
    //     }
    // }
    // ret1.into_shape((h, w)).unwrap()
}

#[cfg(test)]
mod test_mod {
    use super::*;

    #[test]
    fn naive_conv_test() {
        let input_pixels = array![
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ];

        let output_pixels = array![
            [2, 2, 3, 1, 1],
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [0, 2, 2, 1, 1],
        ];

        let kernel = array![[1, 0, 1], [0, 1, 0], [1, 0, 1]];
        // dbg!(input_pixels * output_pixels);
        assert_eq!(conv_2d(&input_pixels, &kernel), output_pixels);
    }
}