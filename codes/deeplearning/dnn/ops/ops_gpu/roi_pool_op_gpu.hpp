#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef ROI_POOL_OP_GPU_HPP
#define ROI_POOL_OP_GPU_HPP

#include <iostream>
#include <cfloat>
#include <cstdio>

#include "cuda_tools.hpp"

void roi_pool_forward_gpu(
    const float* input, const int height, const int width, const int nchannels, const int pooled_height,
    const int pooled_width, const float spatial_scale, const float* rois, const int nrois, float* output,
    int* argmax
);

void roi_pool_backward_gpu(
    const int* argmax, const float* top_grad, const int top_count, float* bottom_grad, const int bottom_count
);

#endif
