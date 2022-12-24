pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

mod conv_2d;

pub use conv_2d::fft::Conv2DFftExt;
pub use conv_2d::Conv2DExt;


pub enum ConvType<const N: usize> {
    Full,
    Same,
    Valid,
    Custom([usize; N], [usize; N]),
    // Custom(&'a [usize], &'a [usize]),
}

// pub enum BorderType<'a, T: num::traits::NumAssign + Copy> {
//     Zero,
//     Const(&'a [T]),
//     Reflect(&'a [bool]),
//     Warp(&'a [bool]),
// }
