#if !GOOGLE_CUDA
#error This file cannot be included without cuda support.
#endif

#ifndef PROPOSAL_ENCODE_OP_GPU_HPP
#define PROPOSAL_ENCODE_OP_GPU_HPP

#include <iostream>
#include <cfloat>
#include "cuda_tools.hpp"

using namespace std;

void proposal_encode_forward_gpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, 
                            const float* shapes, int nrois, float spatial_scale,
                            float* labels, float* targets, int nbatches, int height, int width );

#endif
