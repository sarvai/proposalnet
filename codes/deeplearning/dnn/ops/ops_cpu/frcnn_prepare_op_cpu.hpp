#ifndef FRCNN_ENCODE_OP_CPU_HPP
#define FRCNN_ENCODE_OP_CPU_HPP

#include <iostream>
#include <cmath>

using namespace std;

void frcnn_prepare_forward_cpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                                const float* rois, const float* roilabels, const int* roibatches, int nrois,
                                const float* overlaps,
                                int nclasses, float fg_min_overlap, float bg_max_overlap,
                                float* labels, float* deltas );

#endif
