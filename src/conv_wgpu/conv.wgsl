@group(0)
@binding(0)
var<storage, read> data: array<i32>;
@group(0)
@binding(1)
var<storage, read_write> output: array<i32>;
@group(0)
@binding(2)
var<storage, read> kernal_offset: array<i32>;
@group(0)
@binding(3)
var<storage, read> kernel_value: array<i32>;


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
    var sum = 0;
    var index = global_id.x + global_id.y * 65535 + global_id.z * 65535 * 65535;
    for (var i = 0; i < i32(arrayLength(&kernal_offset)); i++) {
        sum += data[i32(index) + kernal_offset[i]] * kernel_value[i];
    }
    output[index] = sum;

    // data[global_id.x] = i32(global_id.x) * data[global_id.x];
}