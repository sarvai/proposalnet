#if GOOGLE_CUDA

//#define EIGEN_USE_GPU

#include"roi_pool_op_gpu.hpp"

//#define CUDA_1D_KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

__global__ void cuda_fill( const int nthreads, float* arr, const float value )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        arr[ index ] = value;
    }
}

__global__ void ROIPoolForward( const int nthreads, const float* input, const int height, const int width,
                                const int nchannels, const int pooled_height, const int pooled_width,
                                const float spatial_scale, const float* rois, const int nrois, float* output,
                                int* argmax )
{
    int input_base_hwc = height*width*nchannels;
    int input_base_wc = width*nchannels;
    int input_base_c = nchannels;

    int output_base_hwc = pooled_height*pooled_width*nchannels;
    int output_base_wc = pooled_width*nchannels;
    int output_base_c = nchannels;

    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int c = index % output_base_c;
        int pw = (index / (output_base_c)) % pooled_width;
        int ph = (index / (output_base_wc)) % pooled_height;
        int n = (index / (output_base_hwc));

        const float* roi = rois + n * 5;

        int batch_index = static_cast<int>( roi[0] );
        int roi_x0 = static_cast<int>( round(roi[1]*spatial_scale) );
        int roi_y0 = static_cast<int>( round(roi[2]*spatial_scale) );
        int roi_x1 = static_cast<int>( round(roi[3]*spatial_scale) );
        int roi_y1 = static_cast<int>( round(roi[4]*spatial_scale) );

        int roi_width = max( roi_x1 - roi_x0 + 1, 1 );
        int roi_height = max( roi_y1 - roi_y0 + 1, 1 );

        float bin_size_h = static_cast<float>( roi_height ) / static_cast<float>( pooled_height );
        float bin_size_w = static_cast<float>( roi_width ) / static_cast<float>( pooled_width );

        int hstart = roi_y0 + static_cast<int>( floor( ph * bin_size_h ) );
        hstart = min( max( hstart, 0 ), height );

        int hend = roi_y0 + static_cast<int>( ceil( (ph+1) * bin_size_h ) );
        hend = min( max( hend, 0 ), height );

        int wstart = roi_x0 + static_cast<int>(  floor( pw * bin_size_w ) );
        wstart = min( max( wstart, 0 ), width );

        int wend = roi_x0 + static_cast<int>( ceil( (pw+1) * bin_size_w ) );
        wend = min( max( wend, 0 ), width );

        bool is_empty = ( hend <= hstart ) || ( wend <= wstart );

        float maxval = ( is_empty ) ? 0 : -FLT_MAX;
        int maxidx = -1;

        for( int h=hstart ; h<hend ; h++ )
        {
            for( int w=wstart ; w<wend ; w++ )
            {
                int input_index = batch_index*input_base_hwc + h*input_base_wc  + w*input_base_c + c;
                auto v = input[input_index];

                if( v > maxval )
                {
                    maxval = v;
                    maxidx = input_index;
                }
            }
        }

        output[index] = maxval;
        argmax[index] = maxidx;
    }
}


__global__ void ROIPoolBackward( const int nthreads, const int* argmax, const float* top_grad, float* bottom_grad )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int idx = argmax[index];
        atomicAdd( bottom_grad+idx,top_grad[index]);
    }
}

void roi_pool_forward_gpu(
    const float* input, const int height, const int width, const int nchannels, const int pooled_height,
    const int pooled_width, const float spatial_scale, const float* rois, const int nrois, float* output,
    int* argmax
)
{
    const int output_size = nrois * pooled_height * pooled_width * nchannels;
    cudaError_t err;

    int total_blocks = (output_size+kThreadsPerBlock-1)/kThreadsPerBlock;

    ROIPoolForward<<<total_blocks,kThreadsPerBlock>>>(output_size, input, height, width, nchannels, pooled_height, pooled_width, spatial_scale, rois, nrois, output, argmax);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

void roi_pool_backward_gpu(
    const int* argmax, const float* top_grad, const int top_count, float* bottom_grad, const int bottom_count
)
{
    int total_blocks;
    cudaError_t err;

    total_blocks = (bottom_count+kThreadsPerBlock-1)/kThreadsPerBlock;
    cuda_fill<<<total_blocks,kThreadsPerBlock>>>(bottom_count,bottom_grad,0.0);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    total_blocks = (top_count+kThreadsPerBlock-1)/kThreadsPerBlock;
    ROIPoolBackward<<<total_blocks,kThreadsPerBlock>>>(top_count,argmax,top_grad,bottom_grad);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif
