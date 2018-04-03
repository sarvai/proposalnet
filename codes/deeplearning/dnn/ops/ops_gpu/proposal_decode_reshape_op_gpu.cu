#if GOOGLE_CUDA

#include"proposal_decode_reshape_op_gpu.hpp"

__global__ void ProposalDecodeReshapeForward( const int nthreads, const float* input, const int height, const int width,
                                        const int base, const float hbase, float* output )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int grid_w = index % width;
        int grid_h = (index / width) % height;
        int grid_n = (index / (height*width));

        float grid_center_x = grid_w * base + hbase;
        float grid_center_y = grid_h * base + hbase;

        float coded_x = input[ index*4 ];
        float coded_y = input[ index*4+1 ];
        float coded_logw = input[ index*4+2 ];
        float coded_logh = input[ index*4+3 ];

        float w = exp( coded_logw );
        float h = exp( coded_logh );
        float cx = w * coded_x + grid_center_x;
        float cy = h * coded_y + grid_center_y;

        output[ index*5 ] = grid_n;
        output[ index*5+1 ] = cx - w*0.5;
        output[ index*5+2 ] = cy - h*0.5;
        output[ index*5+3 ] = cx + w*0.5;
        output[ index*5+4 ] = cy + h*0.5;
    }
}

void proposal_decode_reshape_forward_gpu( const float* input, int nbatches, int height, int width, int nchannels,
                double spatial_scale, float* output )
{
    int base = round( 1.0 / spatial_scale );
    int total = nbatches * height * width;
    float hbase = static_cast<float>(base)/2;

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    ProposalDecodeReshapeForward<<<total_blocks,kThreadsPerBlock>>>(total, input, height, width, base, hbase, output );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif
