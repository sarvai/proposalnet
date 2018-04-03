#if GOOGLE_CUDA

#include"bbox_delta_gpu.hpp"

__global__ void BboxDeltaInvForward( const float* rois, const float* deltas, int nthreads, float* output )
{
  CUDA_1D_KERNEL_LOOP( index, nthreads )
  {
    int ind = index*4;

    float rx0 = rois[ ind+0 ];
    float ry0 = rois[ ind+1 ];
    float rx1 = rois[ ind+2 ];
    float ry1 = rois[ ind+3 ];

    float rw = rx1 - rx0;
    float rh = ry1 - ry0;
    float rx = rx0 + rw*0.5;
    float ry = ry0 + rh*0.5;

    float dx = deltas[ ind+0 ];
    float dy = deltas[ ind+1 ];
    float dw = deltas[ ind+2 ];
    float dh = deltas[ ind+3 ];

    dw = exp( dw );
    dh = exp( dh );

    float tw = dw * rw;
    float th = dh * rh;
    float tx = dx * rw + rx;
    float ty = dy * rh + ry;

    output[ ind+0 ] = tx - tw*0.5;
    output[ ind+1 ] = ty - th*0.5;
    output[ ind+2 ] = tx + tw*0.5;
    output[ ind+3 ] = ty + th*0.5;
  }
}

void bbox_delta_inv_forward_gpu( const float* rois, const float* deltas, int count, float* output )
{
  cudaError_t err;
  int total_blocks = (count+kThreadsPerBlock-1)/kThreadsPerBlock;

  BboxDeltaInvForward<<<total_blocks,kThreadsPerBlock>>>(rois,deltas,count,output);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }
}

#endif
