#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef FRCNN_PREPARE_OP_GPU_HPP
#define FRCNN_PREPARE_OP_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

void frcnn_prepare_forward_gpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                                const float* rois, const float* roilabels, const int* roibatches, int nrois,
                                const float* overlaps,
                                int nclasses, float fg_min_overlap, float bg_max_overlap,
                                float* labels, float* deltas );

#endif
