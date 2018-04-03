#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef OVERLAPS_OP_GPU_HPP
#define OVERLAPS_OP_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

void overlaps_forward_gpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                                const float* rois, const float* roilabels, const int* roibatches, int nrois,
                                float* overlaps );

#endif
