#ifndef BBOX_DELTA_HPP
#define BBOX_DELTA_HPP

#include <iostream>
#include <cmath>

using namespace std;

void bbox_delta_inv_forward_cpu( const float* rois, const float* deltas, int count, float* output );

#endif
