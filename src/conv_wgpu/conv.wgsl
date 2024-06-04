@group(0)
@binding(0)
var<storage, read> data: array<i32>;
@group(0)
@binding(1)
var<storage, read_write> output: array<i32>;
@group(0)
@binding(2)
var<storage, read> strides: array<vec2<u32>>;

@group(1)
@binding(0)
var<storage, read> kernal_offset: array<i32>;
@group(1)
@binding(1)
var<storage, read> kernel_value: array<i32>;


var<workgroup> item_sum: atomic<i32>;
var<workgroup> index_atomic: u32;
var<workgroup> si_atomic: u32;
const workgroup_len: u32 = 32;

@compute
@workgroup_size(1)
fn test(@builtin(global_invocation_id) workgroup_id: vec3<u32>,
@builtin(num_workgroups) num_workgroups: vec3<u32>,
@builtin(local_invocation_index) local_invocation_index: u32
) {
    var sum = 0;

    let si = workgroup_id.x + workgroup_id.y * 4000;
    let index = si + (si / 4000) * (4040 - 4000);

    for (var i = 0; i < i32(arrayLength(&kernal_offset)); i ++) {
        sum += data[i32(index) + kernal_offset[i]] * kernel_value[i];
    }

    sum += i32(local_invocation_index);
    
    output[si] = sum;
}

@compute
@workgroup_size(1)
fn ttest(@builtin(global_invocation_id) workgroup_id: vec3<u32>,
@builtin(num_workgroups) num_workgroups: vec3<u32>,
@builtin(local_invocation_index) local_invocation_index: u32
) {
    var sum = 0;

    let si = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    let index = si + (si / 4000) * (4040 - 4000);

    for (var i = 0; i < i32(arrayLength(&kernal_offset)); i ++) {
        sum += data[i32(index) + kernal_offset[i]] * kernel_value[i];
    }

    sum += i32(local_invocation_index);
    
    output[si] = sum;
}

@compute
@workgroup_size(workgroup_len)
fn _main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
@builtin(num_workgroups) num_workgroups: vec3<u32>,
@builtin(local_invocation_index) local_invocation_index: u32
) {
    var sum = 0;

    let si = workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let index = si + (si / 4000) * (4040 - 4000);

    for (var i = i32(local_invocation_index); i < i32(arrayLength(&kernal_offset)); i += i32(workgroup_len)) {
        sum += data[i32(index) + kernal_offset[i]] * kernel_value[i];
    }

    if (local_invocation_index != workgroup_len - 1) {
        atomicAdd(&item_sum, sum);
    } 
    workgroupBarrier();

    if (local_invocation_index == workgroup_len - 1){
        output[si] = sum + item_sum;
    }
}

@compute
@workgroup_size(workgroup_len)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
@builtin(num_workgroups) num_workgroups: vec3<u32>,
@builtin(local_invocation_index) local_invocation_index: u32
) {
    var sum = 0;

    var si = 0u;
    var index = 0u;

    if (local_invocation_index == 0) {
        var si = workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y;
        si_atomic = si;
        // index_atomic = si_atomic + (si_atomic / 4000) * (4040 - 4000);
        var addition = 0u;
        for (var i = 0u; i < arrayLength(&strides); i += 1u) {
            addition += (si / strides[i][0]) * strides[i][1];
            si = si % strides[i][0];
        }
        index_atomic = addition;
    }
    workgroupBarrier();

    index = index_atomic;
    si = si_atomic;

    for (var i = local_invocation_index; i < arrayLength(&kernal_offset); i += workgroup_len) {
        sum += data[i32(index) + kernal_offset[i]] * kernel_value[i];
    }

    if (local_invocation_index != workgroup_len - 1) {
        atomicAdd(&item_sum, sum);
    } 
    workgroupBarrier();

    if (local_invocation_index == workgroup_len - 1){
        output[si] = sum + item_sum;
    }
}