#if GOOGLE_CUDA
#include "smooth_L1_gpu.hpp"

__global__ void SmoothL1Forward( const float* diffs, float* outputs, float sigma2, int ndata )
{
	CUDA_1D_KERNEL_LOOP( i, ndata )
	{
		float val = diffs[i];
		float abs_val = abs(val);
		if(  abs_val < 1.0/sigma2 )
			outputs[i] = 0.5*val*val*sigma2;
		else
			outputs[i] = abs_val - 0.5/sigma2;
	}
}

__global__ void SmoothL1Backward( const float* diffs, const float* top_grad, float* bottom_grad, float sigma2, int ndata )
{
	CUDA_1D_KERNEL_LOOP( i, ndata )
	{
		float val = diffs[i];
		float abs_val = abs(val);

		if( abs_val < 1.0/sigma2 )
		{
			bottom_grad[i] = val * sigma2;
		}
		else
		{
			bottom_grad[i] = (float(0) < val) - (val < float(0));
		}
		bottom_grad[i] *= top_grad[i];
	}
}

void smooth_l1_forward_gpu( const float* diffs, float* output, float sigma2, int ndata )
{
	int total_blocks = (ndata+kThreadsPerBlock-1)/kThreadsPerBlock;
	SmoothL1Forward<<<total_blocks,kThreadsPerBlock>>>( diffs, output, sigma2, ndata );
    cudaError_t err;
	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

void smooth_l1_backward_gpu( const float* diffs, const float* top_grad, float* bottom_grad, float sigma2, int ndata )
{
	int total_blocks = (ndata+kThreadsPerBlock-1)/kThreadsPerBlock;
	SmoothL1Backward<<<total_blocks,kThreadsPerBlock>>>( diffs, top_grad, bottom_grad, sigma2, ndata );
    cudaError_t err;
	err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}


#endif
