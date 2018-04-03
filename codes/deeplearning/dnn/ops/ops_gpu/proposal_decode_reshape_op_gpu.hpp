#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef PROPOSAL_DECODE_RESHAPE_OP_GPU_HPP
#define PROPOSAL_DECODE_RESHAPE_OP_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

void proposal_decode_reshape_forward_gpu( const float* input, int nbatches, int height, int width, int nchannels,
                 double spatial_scale, float* output );

#endif
