#if GOOGLE_CUDA

#include "overlaps_op_gpu.hpp"

__global__ void BuildOverlaps( int nthreads, const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                                const float* rois, const float* roilabels, const int* roibatches, int nrois,
                                float* overlaps )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int gtbox_index = index % ngtboxes;
        int roi_index = index / ngtboxes;

        bool label_check = gtlabels[ gtbox_index ] == roilabels[ roi_index ];
        bool batch_check = gtbatches[ gtbox_index ] == roibatches[ roi_index ];

        if( label_check && batch_check )
        {
            const float* box = gtboxes + 4*gtbox_index;
            const float* roi = rois + 4*roi_index;

            float box_area = (box[2]-box[0]+1)*(box[3]-box[1]+1);
            float roi_area = (roi[2]-roi[0]+1)*(roi[3]-roi[1]+1);

            float ix = min( roi[2], box[2] ) - max( roi[0], box[0] );
            float iy = min( roi[3], box[3] ) - max( roi[1], box[1] );
            float intersection = max(0.0,static_cast<double>(ix)+1)*max(0.0,static_cast<double>(iy)+1);

            overlaps[ index ] = intersection / ( box_area + roi_area - intersection );
        }
        else
        {
            overlaps[ index ] = -1;
        }
    }
}

float* build_overlaps( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                       const float* rois, const float* roilabels, const int* roibatches, int nrois )
{
    int total = nrois * ngtboxes;
    float* overlaps;
    cudaMalloc((void**)&overlaps, total * sizeof(float) );

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    BuildOverlaps<<<total_blocks,kThreadsPerBlock>>>( total, gtboxes, gtlabels, gtbatches, ngtboxes,
        rois, roilabels, roibatches, nrois,
        overlaps );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return overlaps;
}

void overlaps_forward_gpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                                const float* rois, const float* roilabels, const int* roibatches, int nrois,
                                float* overlaps )
{
    int total = nrois * ngtboxes;

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    BuildOverlaps<<<total_blocks,kThreadsPerBlock>>>( total, gtboxes, gtlabels, gtbatches, ngtboxes,
        rois, roilabels, roibatches, nrois,
        overlaps );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif
