#ifndef PROPOSAL_ENCODE_OP_CPU_HPP
#define PROPOSAL_ENCODE_OP_CPU_HPP

#include <iostream>
#include <cmath>

using namespace std;

void proposal_encode_forward_cpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, 
                            const float* shapes, int nrois, float spatial_scale,
                            float* labels, float* targets, int nbatches, int height, int width );

#endif