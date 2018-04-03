#ifndef OVERLAPS_OP_CPU_HPP
#define OVERLAPS_OP_CPU_HPP

#include <iostream>
#include <cmath>

using namespace std;

void overlaps_forward_cpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                           const float* rois, const float* roilabels, const int* roibatches, int nrois,
                           float* overlaps );

#endif
