#include "smooth_L1_cpu.hpp"

void smooth_l1_forward_cpu( const float* diffs, float* outputs, float sigma2, int ndata )
{
	for( int i=0 ; i<ndata; i++ )
	{
		if( abs( diffs[i] ) < 1.0/sigma2 )
			outputs[i] = 0.5*diffs[i]*diffs[i]*sigma2;
		else
			outputs[i] = abs( diffs[i] ) - 0.5/sigma2;
	}
}

void smooth_l1_backward_cpu( const float* diffs, const float* top_grad, float* bottom_grad, float sigma2, int ndata )
{
	for( int i=0 ; i<ndata ; i++ )
	{
		if( abs( diffs[i] ) < 1.0/sigma2 )
			bottom_grad[i] = diffs[i] * sigma2;
		else
			bottom_grad[i] = (float(0) < diffs[i]) - (diffs[i] < float(0));
		bottom_grad[i] *= top_grad[i];
	}
}
