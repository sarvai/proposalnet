#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef SMOOTH_L1_GPU_HPP
#define SMOOTH_L1_GPU_HPP

#include <iostream>
#include <cfloat>
#include <cstdio>

#include "cuda_tools.hpp"

void smooth_l1_forward_gpu( const float* diffs, float* outputs, float sigma2, int ndata );
void smooth_l1_backward_gpu( const float* diffs, const float* top_grad, float* bottom_grad, float sigma2, int ndata );

#endif
