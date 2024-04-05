fn good_size_cc(n: usize) -> usize {
    let mut best_fac = n.next_power_of_two();

    loop {
        let new_fac = best_fac / 4 * 3;
        match new_fac.cmp(&n) {
            std::cmp::Ordering::Less => break,
            std::cmp::Ordering::Equal => return n,
            std::cmp::Ordering::Greater => {
                best_fac = new_fac;
            }
        }
    }
    loop {
        let new_fac = best_fac / 6 * 5;
        match new_fac.cmp(&n) {
            std::cmp::Ordering::Less => break,
            std::cmp::Ordering::Equal => return n,
            std::cmp::Ordering::Greater => {
                best_fac = new_fac;
            }
        }
    }

    best_fac
}

fn good_size_rr(n: usize) -> usize {
    let res = n % 2;
    let n = n / 2;

    (good_size_cc(n)) * 2 + res
}

pub fn compute<const N: usize>(size: &[usize; N]) -> [usize; N] {
    std::array::from_fn(|i| {
        if i == N - 1 {
            good_size_rr(size[i])
        } else {
            good_size_cc(size[i])
        }
    })
}
