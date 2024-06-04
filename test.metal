// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
    uint size2;
    uint size3;
    uint size4;
};

typedef int type_1[1];
typedef metal::uint2 type_3[1];
constant uint workgroup_len = 32u;

struct testInput {
};
kernel void test(
  metal::uint3 workgroup_id [[thread_position_in_grid]]
, metal::uint3 num_workgroups [[threadgroups_per_grid]]
, uint local_invocation_index [[thread_index_in_threadgroup]]
, device type_1 const& data [[user(fake0)]]
, device type_1& output [[user(fake0)]]
, device type_1 const& kernal_offset [[user(fake0)]]
, device type_1 const& kernel_value [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    int sum = 0;
    int i = 0;
    uint si_1 = workgroup_id.x + (workgroup_id.y * 4000u);
    uint index_1 = si_1 + ((si_1 / 4000u) * 40u);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            int _e39 = i;
            i = _e39 + 1;
        }
        loop_init = false;
        int _e17 = i;
        if (_e17 < static_cast<int>(1 + (_buffer_sizes.size2 - 0 - 4) / 4)) {
        } else {
            break;
        }
        {
            int _e25 = i;
            int _e27 = kernal_offset[_e25];
            int _e30 = data[static_cast<int>(index_1) + _e27];
            int _e32 = i;
            int _e34 = kernel_value[_e32];
            int _e36 = sum;
            sum = _e36 + (_e30 * _e34);
        }
    }
    int _e42 = sum;
    sum = _e42 + static_cast<int>(local_invocation_index);
    int _e46 = sum;
    output[si_1] = _e46;
    return;
}


struct ttestInput {
};
kernel void ttest(
  metal::uint3 workgroup_id_1 [[thread_position_in_grid]]
, metal::uint3 num_workgroups_1 [[threadgroups_per_grid]]
, uint local_invocation_index_1 [[thread_index_in_threadgroup]]
, device type_1 const& data [[user(fake0)]]
, device type_1& output [[user(fake0)]]
, device type_1 const& kernal_offset [[user(fake0)]]
, device type_1 const& kernel_value [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    int sum_1 = 0;
    int i_1 = 0;
    uint si_2 = workgroup_id_1.x + (workgroup_id_1.y * num_workgroups_1.x);
    uint index_2 = si_2 + ((si_2 / 4000u) * 40u);
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            int _e39 = i_1;
            i_1 = _e39 + 1;
        }
        loop_init_1 = false;
        int _e17 = i_1;
        if (_e17 < static_cast<int>(1 + (_buffer_sizes.size2 - 0 - 4) / 4)) {
        } else {
            break;
        }
        {
            int _e25 = i_1;
            int _e27 = kernal_offset[_e25];
            int _e30 = data[static_cast<int>(index_2) + _e27];
            int _e32 = i_1;
            int _e34 = kernel_value[_e32];
            int _e36 = sum_1;
            sum_1 = _e36 + (_e30 * _e34);
        }
    }
    int _e42 = sum_1;
    sum_1 = _e42 + static_cast<int>(local_invocation_index_1);
    int _e46 = sum_1;
    output[si_2] = _e46;
    return;
}


struct _mainInput {
};
kernel void _main(
  metal::uint3 workgroup_id_2 [[threadgroup_position_in_grid]]
, metal::uint3 num_workgroups_2 [[threadgroups_per_grid]]
, uint local_invocation_index_2 [[thread_index_in_threadgroup]]
, metal::uint3 __local_invocation_id [[thread_position_in_threadgroup]]
, device type_1 const& data [[user(fake0)]]
, device type_1& output [[user(fake0)]]
, device type_1 const& kernal_offset [[user(fake0)]]
, device type_1 const& kernel_value [[user(fake0)]]
, threadgroup metal::atomic_int& item_sum
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (metal::all(__local_invocation_id == metal::uint3(0u))) {
        metal::atomic_store_explicit(&item_sum, 0, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int sum_2 = 0;
    int i_2 = {};
    uint si_3 = (workgroup_id_2.x + (workgroup_id_2.y * num_workgroups_2.x)) + ((workgroup_id_2.z * num_workgroups_2.x) * num_workgroups_2.y);
    uint index_3 = si_3 + ((si_3 / 4000u) * 40u);
    i_2 = static_cast<int>(local_invocation_index_2);
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            int _e45 = i_2;
            i_2 = _e45 + 32;
        }
        loop_init_2 = false;
        int _e23 = i_2;
        if (_e23 < static_cast<int>(1 + (_buffer_sizes.size2 - 0 - 4) / 4)) {
        } else {
            break;
        }
        {
            int _e31 = i_2;
            int _e33 = kernal_offset[_e31];
            int _e36 = data[static_cast<int>(index_3) + _e33];
            int _e38 = i_2;
            int _e40 = kernel_value[_e38];
            int _e42 = sum_2;
            sum_2 = _e42 + (_e36 * _e40);
        }
    }
    if (local_invocation_index_2 != 31u) {
        int _e50 = sum_2;
        int _e51 = metal::atomic_fetch_add_explicit(&item_sum, _e50, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (local_invocation_index_2 == 31u) {
        int _e56 = sum_2;
        int _e58 = metal::atomic_load_explicit(&item_sum, metal::memory_order_relaxed);
        output[si_3] = _e56 + _e58;
        return;
    } else {
        return;
    }
}


struct main_Input {
};
kernel void main_(
  metal::uint3 workgroup_id_3 [[threadgroup_position_in_grid]]
, metal::uint3 num_workgroups_3 [[threadgroups_per_grid]]
, uint local_invocation_index_3 [[thread_index_in_threadgroup]]
, metal::uint3 __local_invocation_id [[thread_position_in_threadgroup]]
, device type_1 const& data [[user(fake0)]]
, device type_1& output [[user(fake0)]]
, device type_1 const& kernal_offset [[user(fake0)]]
, device type_1 const& kernel_value [[user(fake0)]]
, threadgroup metal::atomic_int& item_sum
, threadgroup uint& item_index
, threadgroup uint& item_si
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (metal::all(__local_invocation_id == metal::uint3(0u))) {
        metal::atomic_store_explicit(&item_sum, 0, metal::memory_order_relaxed);
        item_index = {};
        item_si = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int sum_3 = 0;
    uint si = 0u;
    uint index = 0u;
    uint i_3 = {};
    if (local_invocation_index_3 == 0u) {
        item_si = (workgroup_id_3.x + (workgroup_id_3.y * num_workgroups_3.x)) + ((workgroup_id_3.z * num_workgroups_3.x) * num_workgroups_3.y);
        uint _e25 = item_si;
        uint _e27 = item_si;
        item_index = _e25 + ((_e27 / 4000u) * 40u);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint _e34 = item_index;
    index = _e34;
    uint _e36 = item_si;
    si = _e36;
    i_3 = local_invocation_index_3;
    bool loop_init_3 = true;
    while(true) {
        if (!loop_init_3) {
            uint _e60 = i_3;
            i_3 = _e60 + workgroup_len;
        }
        loop_init_3 = false;
        uint _e38 = i_3;
        if (_e38 < (1 + (_buffer_sizes.size2 - 0 - 4) / 4)) {
        } else {
            break;
        }
        {
            uint _e43 = index;
            uint _e46 = i_3;
            int _e48 = kernal_offset[_e46];
            int _e51 = data[static_cast<int>(_e43) + _e48];
            uint _e53 = i_3;
            int _e55 = kernel_value[_e53];
            int _e57 = sum_3;
            sum_3 = _e57 + (_e51 * _e55);
        }
    }
    if (local_invocation_index_3 != 31u) {
        int _e65 = sum_3;
        int _e66 = metal::atomic_fetch_add_explicit(&item_sum, _e65, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (local_invocation_index_3 == 31u) {
        uint _e70 = si;
        int _e72 = sum_3;
        int _e74 = metal::atomic_load_explicit(&item_sum, metal::memory_order_relaxed);
        output[_e70] = _e72 + _e74;
        return;
    } else {
        return;
    }
}
