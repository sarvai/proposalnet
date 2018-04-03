#ifndef ROI_POOL_OP_CPU_HPP
#define ROI_POOL_OP_CPU_HPP

#include <cfloat>
#include <cmath>
#include <iostream>

void roi_pool_forward_cpu(
    const float* input, const int height, const int width, const int nchannels, const int pooled_height,
    const int pooled_width, const float spatial_scale, const float* rois, const int nrois, float* output,
    int* argmax
);

void roi_pool_backward_cpu(
    const int* argmax, const float* top_grad, const int top_count, float* bottom_grad, const int bottom_count
);

#endif
