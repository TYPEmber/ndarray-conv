// Example comparing standard convolution and FFT-accelerated convolution
// Tests with data shape 2049x2049 and kernel shape 7x7
//
// Without parallelism:  cargo run --example compare_conv_methods --release
// With rayon:           cargo run --example compare_conv_methods --release --features rayon

use ndarray::Array2;
use ndarray_conv::{get_fft_processor, ConvExt, ConvFFTExt, ConvMode, PaddingMode};
use std::time::{Duration, Instant};

fn elapsed<F: FnOnce() -> T, T>(f: F) -> (T, Duration) {
    let t = Instant::now();
    let v = f();
    (v, t.elapsed())
}

fn fmt_dur(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.1}ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.1}us", d.as_secs_f64() * 1e6)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Convolution Method Comparison");
    #[cfg(feature = "rayon")]
    println!("(rayon feature enabled)");
    #[cfg(not(feature = "rayon"))]
    println!("(rayon feature disabled -- rerun with --features rayon for par variants)");
    println!("{}", "=".repeat(80));
    println!();

    let data = create_test_array_f32(2049, 2049);
    let kernel = create_test_array_f32(7, 7);
    println!("Input: 2049x2049   Kernel: 7x7   Mode: Same / Replicate");
    println!();
    println!("{}", "-".repeat(80));

    // 1. conv
    let (result_conv, dur_conv) = elapsed(|| {
        data.conv(&kernel, ConvMode::Same, PaddingMode::Replicate)
            .unwrap()
    });

    // 2. conv_fft
    let (result_fft, dur_fft) = elapsed(|| {
        data.conv_fft(&kernel, ConvMode::Same, PaddingMode::Replicate)
            .unwrap()
    });

    // 3. conv_fft_with_processor
    let mut proc = get_fft_processor::<f32, f32>();
    let (result_fft_proc, dur_fft_proc) = elapsed(|| {
        data.conv_fft_with_processor(&kernel, ConvMode::Same, PaddingMode::Replicate, &mut proc)
            .unwrap()
    });

    // 4. conv_fft_par  (rayon only)
    #[cfg(feature = "rayon")]
    let (result_fft_par, dur_fft_par) = elapsed(|| {
        data.conv_fft_par(&kernel, ConvMode::Same, PaddingMode::Replicate)
            .unwrap()
    });



    let max_diff = |a: &Array2<f32>, b: &Array2<f32>| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mean_diff = |a: &Array2<f32>, b: &Array2<f32>| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32
    };

    println!("{:<40} {:>12}   max_diff     mean_diff", "method", "time");
    println!("{}", "-".repeat(80));
    println!("{:<40} {:>12}", "conv (baseline)", fmt_dur(dur_conv));
    println!(
        "{:<40} {:>12}   {:.3e}   {:.3e}",
        "conv_fft",
        fmt_dur(dur_fft),
        max_diff(&result_conv, &result_fft),
        mean_diff(&result_conv, &result_fft)
    );
    println!(
        "{:<40} {:>12}   {:.3e}   {:.3e}",
        "conv_fft_with_processor",
        fmt_dur(dur_fft_proc),
        max_diff(&result_conv, &result_fft_proc),
        mean_diff(&result_conv, &result_fft_proc)
    );
    #[cfg(feature = "rayon")]
    println!(
        "{:<40} {:>12}   {:.3e}   {:.3e}",
        "conv_fft_par",
        fmt_dur(dur_fft_par),
        max_diff(&result_conv, &result_fft_par),
        mean_diff(&result_conv, &result_fft_par)
    );

    println!();

    println!("{}", "=".repeat(80));
    println!("Testing Different Convolution Modes (Valid, Full)");
    println!("{}", "=".repeat(80));
    println!();
    test_conv_modes(&data, &kernel)?;

    Ok(())
}

fn create_test_array_f32(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i + j) % 256) as f32)
}

fn test_conv_modes(
    data: &Array2<f32>,
    kernel: &Array2<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let max_diff = |a: &Array2<f32>, b: &Array2<f32>| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(f32::NEG_INFINITY, f32::max)
    };

    for (mode_name, mode) in [("Valid", ConvMode::Valid), ("Full", ConvMode::Full)] {
        println!("-- Mode: {} --", mode_name);

        let (result_conv, dur_conv) =
            elapsed(|| data.conv(kernel, mode, PaddingMode::Replicate).unwrap());
        let (result_fft, dur_fft) =
            elapsed(|| data.conv_fft(kernel, mode, PaddingMode::Replicate).unwrap());
        let mut proc = get_fft_processor::<f32, f32>();
        let (result_fft_proc, dur_fft_proc) = elapsed(|| {
            data.conv_fft_with_processor(kernel, mode, PaddingMode::Replicate, &mut proc)
                .unwrap()
        });
        #[cfg(feature = "rayon")]
        let (result_fft_par, dur_fft_par) = elapsed(|| {
            data.conv_fft_par(kernel, mode, PaddingMode::Replicate)
                .unwrap()
        });


        println!("   output: {}x{}", result_conv.nrows(), result_conv.ncols());
        println!("{}", "-".repeat(60));
        println!("{:<35} {:>10}   max_diff_vs_conv", "method", "time");
        println!("{}", "-".repeat(60));
        println!("{:<35} {:>10}", "conv (baseline)", fmt_dur(dur_conv));
        println!(
            "{:<35} {:>10}   {:.3e}",
            "conv_fft",
            fmt_dur(dur_fft),
            max_diff(&result_conv, &result_fft)
        );
        println!(
            "{:<35} {:>10}   {:.3e}",
            "conv_fft_with_processor",
            fmt_dur(dur_fft_proc),
            max_diff(&result_conv, &result_fft_proc)
        );
        #[cfg(feature = "rayon")]
        println!(
            "{:<35} {:>10}   {:.3e}",
            "conv_fft_par",
            fmt_dur(dur_fft_par),
            max_diff(&result_conv, &result_fft_par)
        );

        println!();
    }

    Ok(())
}
