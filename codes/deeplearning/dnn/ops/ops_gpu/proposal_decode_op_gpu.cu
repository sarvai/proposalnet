#if GOOGLE_CUDA

#include"proposal_decode_op_gpu.hpp"

__global__ void ProposalDecodeForward( const int nthreads, const float* input, const int height, const int width,
                                        const int base, const float hbase, float* output )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int grid_w = index % width;
        int grid_h = (index / width) % height;
        //int grid_n = (index / (height*width*nclasses));

        float grid_center_x = grid_w * base + hbase;
        float grid_center_y = grid_h * base + hbase;

        float coded_x = input[ index*4 ];
        float coded_y = input[ index*4+1 ];
        float coded_logw = input[ index*4+2 ];
        float coded_logh = input[ index*4+3 ];

        float w = exp( coded_logw );
        if( isnan(w) || isinf(w) )
            w = 0.0;
        float h = exp( coded_logh );
        if( isnan(h) || isinf(h) )
            h = 0.0;
        float cx = w * coded_x + grid_center_x;
        float cy = h * coded_y + grid_center_y;

        output[ index*4 ] = cx - w*0.5;
        output[ index*4+1 ] = cy - h*0.5;
        output[ index*4+2 ] = cx + w*0.5;
        output[ index*4+3 ] = cy + h*0.5;


    }
}

void proposal_decode_forward_gpu( const float* input, int nbatches, int height, int width, int nchannels,
                double spatial_scale, float* output )
{
    int base = round( 1.0 / spatial_scale );
    //int nclasses = nchannels / 4;
    int total = nbatches * height * width;
    float hbase = static_cast<float>(base)/2;

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    ProposalDecodeForward<<<total_blocks,kThreadsPerBlock>>>(total, input, height, width, base, hbase, output );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif
