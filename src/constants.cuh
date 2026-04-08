#pragma once
// includes with defaults BSIZE=1024, IS_LONG=0, XXX_CG_SIZE_LOG=3, XXX_CG_AMOUNT_LOG=2
#include "constants_config.cuh"

//------- for all the RT algorithms -----------
// defines if the BVH should be compacted, needed in rtx_functions.h
#define COMPACT 0

//------- some RTX algorithms have the option to use thread reordering --------
// if == 0 only SER uses thread reordering
#define ThreadReordering 0

//------- for the XXX and Interleaved algorithm -------
constexpr uint32_t XXX_MAX_LEVELS = 20;
constexpr uint32_t XXX_LEVEL_ALIGNMENT = 1u << 5u;

//------- general definitions -------------------------------
template <typename nType, typename n2Type>
struct Params {
  OptixTraversableHandle handle;
  float *output;
  float2 *query;
  n2Type *iquery;
  float *d_array;
  nType *idx_output;
  nType *min_block;
  float min;
  float max;
  nType num_blocks;
  nType block_size;
  nType nb;
};

template <typename nType, typename n2Type>
struct xxx_metadata_type {
    nType num_levels;
    nType level_offset[XXX_MAX_LEVELS];
    nType level_size[XXX_MAX_LEVELS];
};

template <typename nType, typename n2Type>
struct ParamsInterleaved {
  OptixTraversableHandle handle;
  uint32_t q;
  n2Type* query;
  float* output;
  float* array;
  float* level_buffer;
  nType scan_threshold;
  float min;
  float max;
  xxx_metadata_type<nType, n2Type> metadata;
};
