@group(0) @binding(0)
var<storage, read> data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let N = arrayLength(&data);
    let k = global_id.x;
    var sum = vec2<f32>(0.0, 0.0);

    for (var n = 0u; n < N; n = n + 1u) {
        let phase = 2.0 * 3.14159265359 * f32(k) * f32(n) / f32(N);
        sum = sum + vec2<f32>(
            data[n] * cos(phase),
            -data[n] * sin(phase)
        );
    }

    let index = k * 2;
    if (index + 1 < arrayLength(&output)) {
        output[index] = sum.x;
        output[index + 1] = sum.y;
    }
}