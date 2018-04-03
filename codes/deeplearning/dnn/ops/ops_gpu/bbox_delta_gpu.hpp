#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef BBOX_DELTA_GPU_HPP
#define BBOX_DELTA_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

void bbox_delta_inv_forward_gpu( const float* rois, const float* deltas, int count, float* output );

#endif