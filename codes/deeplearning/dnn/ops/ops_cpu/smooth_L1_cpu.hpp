#ifndef SMOOTH_L1_CPU_HPP
#define SMOOTH_L1_CPU_HPP

#include <iostream>
#include <cmath>

using namespace std;

void smooth_l1_forward_cpu( const float* diffs, float* outputs, float sigma2, int ndata );
void smooth_l1_backward_cpu( const float* diffs, const float* top_grad, float* bottom_grad, float sigma2, int ndata );

#endif