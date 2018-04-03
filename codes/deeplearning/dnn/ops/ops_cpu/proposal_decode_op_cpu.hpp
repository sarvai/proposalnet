#ifndef PROPOSAL_DECODE_OP_CPU_HPP
#define PROPOSAL_DECODE_OP_CPU_HPP

#include <iostream>
#include <cmath>

using namespace std;

void proposal_decode_forward_cpu( const float* input, int nbatches, int height, int width, int nchannels,
                 double spatial_scale, float* output );

#endif
