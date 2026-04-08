#include <optix.h>
#include <math.h>
#include <stdint.h>
#include "constants.cuh"

#if IS_LONG
using nType = uint64_t;
using n2Type = ulong2;
#else
using nType = uint32_t;
using n2Type = uint2;
#endif

extern "C" static __constant__ Params<nType, n2Type> params;
extern "C" static __constant__ ParamsInterleaved<nType, n2Type> params_xxx;

extern "C" __global__ void __raygen__rmq_blocks() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;
  nType num_blocks = params.num_blocks;
  nType block_size = params.block_size;

  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;

  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;

  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  float temp_min = max;

  n2Type q = params.iquery[idx.x];
  // here always int64_t as type, since in the code we calculate -1, which does not allow an unsigned data type
  int64_t lB = q.x / block_size;
  int64_t rB = q.y / block_size;

  //printf("Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n", idx.x, q.x, q.y, lB, rB);
  nType bx, by;
  float x, y;
  float3 ray_origin, ray_direction;

#if ThreadReordering == 1
  // Thread reordering
  unsigned int hint;
  if (lB == rB)
    hint = 1;
  else
    hint = 2 + (rB - lB >  1);
  optixReorder(hint, 2);
#endif
  
  if (lB == rB) {
    bx = (lB+1) % num_blocks;
    by = (rB+1) / num_blocks;
    nType mx = q.x % block_size;
    nType my = q.y % block_size;
    x = 2.0f*bx + ((float)mx / block_size);
    y = 2.0f*by + ((float)my / block_size);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    params.output[idx.x] = optixHitObjectGetRayTmax() + min;
    return;
  }

  // search min in fully contained blocks
  if (lB < rB-1) {
    x = (float)(lB+1) / (1<<23);
    y = (float)(rB-1) / (1<<23);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    temp_min = optixHitObjectGetRayTmax();
  }

  // search min in first partial block
  nType mod = q.x % block_size;
  bx = (lB+1) % num_blocks;
  by = (lB+1) / num_blocks;
  x = 2*bx + ((float)mod / block_size);
  y = 2*by+1;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  float t = optixHitObjectGetRayTmax();
  if (temp_min > t) {
    temp_min = t;
  }

  // search min in last partial block
  mod = q.y % block_size;
  bx = (rB+1) % num_blocks;
  by = (rB+1) / num_blocks;
  x = 2*bx;
  y = 2*by + (float)mod / block_size;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  t = optixHitObjectGetRayTmax();
  if (temp_min > t) {
    temp_min = t;
  }

  params.output[idx.x] = temp_min + min;
}

extern "C" __global__ void __raygen__rmq_blocks_idx() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;
  nType num_blocks = params.num_blocks;
  nType block_size = params.block_size;

  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;

  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;

  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  float temp_min = max;
  nType temp_min_idx;

  n2Type q = params.iquery[idx.x];
  // here always int64_t as type, since in the code we calculate -1, which does not allow an unsigned data type
  int64_t lB = q.x / block_size;
  int64_t rB = q.y / block_size;
  
  nType bx, by;
  float x, y;
  float3 ray_origin, ray_direction;

  nType nb_offset = params.nb;

#if ThreadReordering == 1
  // Thread reordering
  unsigned int hint;
  if (lB == rB)
    hint = 1;
  else
    hint = 2 + (rB - lB >  1);
  optixReorder(hint, 2);
#endif
  
  if (lB == rB) {
    bx = (lB+1) % num_blocks;
    by = (rB+1) / num_blocks;
    nType mx = q.x % block_size;
    nType my = q.y % block_size;
    x = 2.0f*bx + ((float)mx / block_size);
    y = 2.0f*by + ((float)my / block_size);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    params.idx_output[idx.x] = optixHitObjectGetPrimitiveIndex() - nb_offset;
    return;
  }

  // search min in fully contained blocks
  if (lB < rB-1) {
    x = (float)(lB+1) / (1<<23);
    y = (float)(rB-1) / (1<<23);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    temp_min_idx = params.min_block[optixHitObjectGetPrimitiveIndex()];
    temp_min = optixHitObjectGetRayTmax() + min;
  }

  // search min in first partial block
  nType mod = q.x % block_size;
  bx = (lB+1) % num_blocks;
  by = (lB+1) / num_blocks;
  x = 2*bx + ((float)mod / block_size);
  y = 2*by+1;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  float t = optixHitObjectGetRayTmax() + min;
  if (temp_min > t) {
    temp_min_idx = optixHitObjectGetPrimitiveIndex() - nb_offset;
    temp_min = t;
  }
    
  // search min in last partial block
  mod = q.y % block_size;
  bx = (rB+1) % num_blocks;
  by = (rB+1) / num_blocks;
  x = 2*bx;
  y = 2*by + (float)mod / block_size;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  t = optixHitObjectGetRayTmax() + min;
  if (temp_min > t) {
    temp_min_idx = optixHitObjectGetPrimitiveIndex() - nb_offset;
    temp_min = t;
  }

  params.idx_output[idx.x] = temp_min_idx;
}

extern "C" __global__ void __raygen__rmq_blocks_ser() {
  const uint3 idx = optixGetLaunchIndex();
  float &min = params.min;
  float &max = params.max;
  nType num_blocks = params.num_blocks;
  nType block_size = params.block_size;


  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;

  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;

  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  float temp_min = max;
  float t;

  n2Type q = params.iquery[idx.x];
  // here always int64_t as type, since in the code we calculate -1, which does not allow an unsigned data type
  int64_t lB = q.x / block_size;
  int64_t rB = q.y / block_size;

  //printf("Ray %i, query (%i,%i)\n    lB = %i,  rB = %i\n", idx.x, q.x, q.y, lB, rB);
  nType bx, by;
  float x, y;
  float3 ray_origin, ray_direction;

  // Thread reordering
  unsigned int hint;
  if (lB == rB)
    hint = 1;
  else
    hint = 2 + (rB - lB >  1);
  optixReorder(hint, 2);
  
  if (lB == rB) {
    bx = (lB+1) % num_blocks;
    by = (rB+1) / num_blocks;
    nType mx = q.x % block_size;
    nType my = q.y % block_size;
    x = 2.0f*bx + ((float)mx / block_size);
    y = 2.0f*by + ((float)my / block_size);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    params.output[idx.x] = optixHitObjectGetRayTmax() + min;
    return;
  }

  // search min in fully contained blocks
  if (lB < rB-1) {
    x = (float)(lB+1) / (1<<23);
    y = (float)(rB-1) / (1<<23);
    ray_origin = make_float3(min, x, y);
    ray_direction = make_float3(1.0, 0.0, 0.0);
    optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
    temp_min = optixHitObjectGetRayTmax();
  }

  // search min in first partial block
  nType mod = q.x % block_size;
  bx = (lB+1) % num_blocks;
  by = (lB+1) / num_blocks;
  x = 2*bx + ((float)mod / block_size);
  y = 2*by+1;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  t = optixHitObjectGetRayTmax();
  if (temp_min > t){
    temp_min = t;
  }

  // search min in last partial block
  mod = q.y % block_size;
  bx = (rB+1) % num_blocks;
  by = (rB+1) / num_blocks;
  x = 2*bx;
  y = 2*by + (float)mod / block_size;
  ray_origin = make_float3(min, x, y);
  optixTraverse(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
  t = optixHitObjectGetRayTmax();
  if (temp_min > t){
    temp_min = t;
  }

  params.output[idx.x] = temp_min + min;
}

// --- kernels for Interleaved and every step between XXX and Interleaved --------------

extern "C" __device__ __forceinline__ 
float atomicMin(float* addr, float value) {
    float old;
    old = __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    return old;
}

template <nType cg_size, nType cg_amount>
//extern "C" 
__device__ __forceinline__
float xxx_static_scan_min(const nType lane_idx, const nType tile_idx, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    constexpr nType alignment = cg_amount * cg_size;
    const nType start = i_left & ~(alignment - 1);
    const nType offset = start + lane_idx;

    float item = local_min;
    #pragma unroll
    for (nType k = 0; k < cg_amount; ++k) {
        nType i = offset + k * cg_size;
        if (i >= i_left && i < e_right) {
          item = min(item, d_array[i]);
        }
    }
    return item;
}

template <nType cg_size>
//extern "C"
__device__ __forceinline__
float xxx_scan_min(const nType tile_idx, const nType lane_idx, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    nType start = i_left & ~(cg_size - 1);

    float item = local_min;
    for (; start < e_right; start += cg_size) {
        nType i = start + lane_idx;
        if (i >= i_left && i < e_right) {
          item = min(item, d_array[i]);
        }
    }
    return item;
}

// Interleaved
extern "C" __global__
void __raygen__interleaved_query() {
  const unsigned int thread_idx = optixGetLaunchIndex().x;
  const unsigned int grid_dim = optixGetLaunchDimensions().x;
  const unsigned int q = params_xxx.q;
  const n2Type* query = params_xxx.query;
  float* output = params_xxx.output;
  float* array = params_xxx.array;
  float* level_buffer = params_xxx.level_buffer;
  const xxx_metadata_type<nType, n2Type> metadata = params_xxx.metadata;
  const int64_t num_levels = metadata.num_levels-1;
  const float &min = params_xxx.min;
  const float &max = params_xxx.max;
  const nType scan_threshold = params_xxx.scan_threshold;

  constexpr nType cg_size = 1u << XXX_CG_SIZE_LOG;
  constexpr nType cg_amount = 1u << XXX_CG_AMOUNT_LOG;

  constexpr nType reduction_factor_log = XXX_CG_SIZE_LOG + XXX_CG_AMOUNT_LOG;
  constexpr nType reduction_factor = 1u << reduction_factor_log;

  const nType tile_idx = thread_idx >> XXX_CG_SIZE_LOG;
  const nType lane_idx = thread_idx % cg_size;
  const nType max_level_reduction_factor_log = reduction_factor_log*num_levels;
  
  // OptiX preparation
  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;

  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;

  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  float x, y;
  float3 ray_origin, ray_direction;
  float local_mins[1 << XXX_CG_SIZE_LOG];
  float local_RT_min;
  n2Type ranges[1 << XXX_CG_SIZE_LOG];

  // skip work queue, since there is no warp shuffles, etc.
  // outer for-loop for the case that grid_dim < q
  for (nType query_idx = tile_idx*cg_size;  query_idx < q; query_idx += grid_dim) {
    local_RT_min = max;
    // initialise the mins and read the RMQs
    for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
      local_mins[active_idx] = max;
      ranges[active_idx] = query[query_idx + active_idx];
    } 

    const nType local_l = ranges[lane_idx].x;
    const nType local_r = ranges[lane_idx].y;
    
    const nType ray_left_idx = (local_l >> (max_level_reduction_factor_log)) + 1;
    // here always int64_t as type, since the -1 does not allow an unsigned data type
    const int64_t ray_right_idx = (local_r >> (max_level_reduction_factor_log)) - 1;

    // if (thread_idx == 0) {
    //   printf("thread_idx: %i, local_l: %u, local_r: %u, ray_left_idx: %i, ray_right_idx: %i\n", thread_idx, local_l, local_r, ray_left_idx, ray_right_idx);
    // }
  
    if (ray_left_idx <= ray_right_idx) {
      x = (float)ray_left_idx / (1<<23);
      y = (float)ray_right_idx / (1<<23);
      ray_origin = make_float3(min, x, y);
      ray_direction = make_float3(1.0, 0.0, 0.0);
      optixTraverse(params_xxx.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
      local_RT_min = optixHitObjectGetRayTmax() + min;
      //printf("thread_idx: %i, RT from left_idx: %u, right_idx: %u, RT result: %f\n", thread_idx, ray_left_idx, ray_right_idx, local_RT_min);
    }

    // hierarchical lookup starts here
    for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
      nType current_left = ranges[active_idx].x;
      // + 1 important, since the hierarchical lookup is implemented with the right bound being exclusive
      nType current_right = ranges[active_idx].y + 1;
      nType current_level = 0;

      for (; current_level < num_levels; ++current_level) {
        if (current_right - current_left <= scan_threshold) {
          break;
        }

        // next multiple of reduction_factor
        nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
        // previous multiple of reduction_factor
        nType prev_right = current_right & ~(reduction_factor - 1);

        // search original array for the first level, otherwise search in the level buffer
        float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];

        local_mins[active_idx] = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_mins[active_idx], base_ptr, current_left, next_left);
        local_mins[active_idx] = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_mins[active_idx], base_ptr, prev_right, current_right);

        //printf("thread_idx: %i, current level: %u, active_idx: %i, local_mins[active_idx]: %f\n", thread_idx, current_level, active_idx, local_mins[active_idx]);
        current_left = next_left >> reduction_factor_log;
        current_right = prev_right >> reduction_factor_log;
      }

      // search original array for the first level, otherwise search in the level buffer
      float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
      local_mins[active_idx] = xxx_scan_min<cg_size>(tile_idx, lane_idx, local_mins[active_idx], base_ptr, current_left, current_right);
    }
    
    local_mins[lane_idx] = fminf(local_mins[lane_idx], local_RT_min);
    for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
      atomicMin(output + query_idx + active_idx, local_mins[active_idx]);
    }
  }
}

// better Interleaved kernel
extern "C" __global__
void __raygen__interleaved_query2() {
  const unsigned int q = params_xxx.q;
  const n2Type* query = params_xxx.query;
  float* output = params_xxx.output;
  float* array = params_xxx.array;
  float* level_buffer = params_xxx.level_buffer;
  const xxx_metadata_type<nType, n2Type> metadata = params_xxx.metadata;
  const int64_t num_levels = metadata.num_levels-1;
  const float &min = params_xxx.min;
  const float &max = params_xxx.max;
  const nType scan_threshold = params_xxx.scan_threshold;

  constexpr nType cg_size = 1u << XXX_CG_SIZE_LOG;
  constexpr nType cg_amount = 1u << XXX_CG_AMOUNT_LOG;

  constexpr nType reduction_factor_log = XXX_CG_SIZE_LOG + XXX_CG_AMOUNT_LOG;
  constexpr nType reduction_factor = 1u << reduction_factor_log;

  const unsigned int thread_idx = optixGetLaunchIndex().x;

  n2Type range;
  const nType tile_idx = thread_idx >> XXX_CG_SIZE_LOG;
  const nType lane_idx = thread_idx % cg_size;
  const unsigned int grid_dim = optixGetLaunchDimensions().x;
  
  // OptiX preparation
  float tmin = 0;
  float tmax = max - min;
  float ray_time = 0;
  OptixVisibilityMask visibilityMask = 255;

  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;

  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTindex = 0;
  float x, y;
  float3 ray_origin, ray_direction;

  const nType max_level_reduction_factor_log = reduction_factor_log*num_levels;

  // skip work queue, since there is no warp shuffles, etc.
  // outer for-loop for the case that grid_dim < q
  for (nType query_idx = tile_idx*cg_size;  query_idx < q; query_idx += grid_dim) {
    // initialise the RT min and read the local RMQ for the RT
    float local_RT_min = max;
    range = query[query_idx + lane_idx];
    
    const nType ray_left_idx = (range.x + (1 << max_level_reduction_factor_log) - 1) >> max_level_reduction_factor_log;
    // here always int64_t as type, since the -1 does not allow an unsigned data type
    const int64_t ray_right_idx = ((range.y + 1) >> (max_level_reduction_factor_log)) -1;
  
    if (ray_left_idx <= ray_right_idx) {
      x = (float)ray_left_idx / (1<<23);
      y = (float)ray_right_idx / (1<<23);
      ray_origin = make_float3(min, x, y);
      ray_direction = make_float3(1.0, 0.0, 0.0);
      optixTraverse(params_xxx.handle, ray_origin, ray_direction, tmin, tmax, ray_time, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
      local_RT_min = optixHitObjectGetRayTmax() + min;
    }

    // hierarchical lookup starts here
    for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
      range = query[query_idx + active_idx];
      float local_min = max;

      // lookup starts here
      {
        nType current_left = range.x;
        // + 1 important, since the hierarchical lookup is implemented with the right bound being exclusive
        nType current_right = range.y + 1;
        nType current_level = 0;

        for (; current_level < num_levels; ++current_level) {
          if (current_right - current_left <= scan_threshold) {
            break;
          }

          // next multiple of reduction_factor
          nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
          // previous multiple of reduction_factor
          nType prev_right = current_right & ~(reduction_factor - 1);

          // search original array for the first level, otherwise search in the level buffer
          float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
          local_min = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, current_left, next_left);
          local_min = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, prev_right, current_right);
          
          // if (thread_idx == 0) {
          //   printf("thread_idx: %i, current level: %u, active_idx: %i, local_min: %f, current_left: %i, next_left: %i, prev_right: %i, current_right: %i\n", thread_idx, current_level, active_idx, local_min, current_left, next_left, prev_right, current_right);
          // }
          current_left = next_left >> reduction_factor_log;
          current_right = prev_right >> reduction_factor_log;
        }

        // search original array for the first level, otherwise search in the level buffer
        if (current_level < num_levels) {
          float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
          local_min = xxx_scan_min<cg_size>(tile_idx, lane_idx, local_min, base_ptr, current_left, current_right);
        }
      }

      if (active_idx == lane_idx) {
        local_min = fminf(local_min, local_RT_min);
      }
      atomicMin(output + query_idx + active_idx, local_min);
    }
  }
}

// Interleaved version without RT
extern "C" __global__
void __raygen__interleaved_without_RT() {
  const unsigned int q = params_xxx.q;
  const n2Type* query = params_xxx.query;
  float* output = params_xxx.output;
  float* array = params_xxx.array;
  float* level_buffer = params_xxx.level_buffer;
  const xxx_metadata_type<nType, n2Type> metadata = params_xxx.metadata;
  const float &max = params_xxx.max;

  constexpr nType cg_size = 1u << XXX_CG_SIZE_LOG;
  constexpr nType cg_amount = 1u << XXX_CG_AMOUNT_LOG;

  constexpr nType reduction_factor_log = XXX_CG_SIZE_LOG + XXX_CG_AMOUNT_LOG;
  constexpr nType reduction_factor = 1u << reduction_factor_log;

  const unsigned int thread_idx = optixGetLaunchIndex().x;
  
  n2Type range;
  const nType tile_idx = thread_idx >> XXX_CG_SIZE_LOG;
  const nType lane_idx = thread_idx % cg_size;
  const unsigned int grid_dim = optixGetLaunchDimensions().x;

  // skip work queue, since there is no warp shuffles, etc.
  // outer for-loop for the case that grid_dim < q
  for (nType query_idx = tile_idx*cg_size;  query_idx < q; query_idx += grid_dim) {
    // skip the RT part here

    // hierarchical lookup starts here
    for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
      range = query[query_idx + active_idx];  
      float local_min = max;
      
      {
        nType current_left = range.x;
        // + 1 important, since the hierarchical lookup is implemented with the right bound being exclusive
        nType current_right = range.y + 1;
        nType current_level = 0;

        for (; current_level < metadata.num_levels - 1; ++current_level) {
            if (current_right - current_left <= 2*reduction_factor) {
                break;
            }

            // next multiple of reduction_factor
            nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
            // previous multiple of reduction_factor
            nType prev_right = current_right & ~(reduction_factor - 1);

            // search original array for the first level, otherwise search in the level buffer
            const float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
            local_min = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, current_left, next_left);
            local_min = xxx_static_scan_min<cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, prev_right, current_right);

            //printf("thread_idx: %i, current level: %u, active_idx: %i\n", thread_idx, current_level, active_idx);
            current_left = next_left >> reduction_factor_log;
            current_right = prev_right >> reduction_factor_log;
        }

        // search original array for the first level, otherwise search in the level buffer
        const float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
        local_min = xxx_scan_min<cg_size>(tile_idx, lane_idx, local_min, base_ptr, current_left, current_right);
        //printf("thread_idx: %i, query idx: %i, active_idx: %i, local_min: %f, lane_idx: %i\n", thread_idx, query_idx, active_idx, local_min, lane_idx);

        // skipped the fminf, since we here do not have a RT min
        atomicMin(output + query_idx + active_idx, local_min);
      }
    }
  } 
}
