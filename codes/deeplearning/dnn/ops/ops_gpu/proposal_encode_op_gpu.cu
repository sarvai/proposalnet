#if GOOGLE_CUDA

#include "proposal_encode_op_gpu.hpp"

__global__ void ProposalEncodeFill( int nthreads, float* ptr, float value )
{
    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        ptr[index] = value;
    }
}

__global__ void ProposalEncodeForward( int nthreads, const float* gtboxes, const float* gtlabels, const int* gtbatches,
                                        const float* shapes, int nrois, float spatial_scale, float* labels, float* targets,
                                        int nbatches, int height, int width )
{
    int base = round( 1.0/spatial_scale );
    float hbase = static_cast<float>( base ) / 2.0; // Half base

    CUDA_1D_KERNEL_LOOP( index, nthreads )
    {
        int w = index % width;
        int h = (index/width) % height;
        int b = index/(width*height);

        const float* shape = shapes + b*2;
        int batch_height = shape[0];
        int batch_width = shape[1];

        int cx = w * base + hbase;
        int cy = h * base + hbase;

        float gx0 = cx - hbase;
        float gy0 = cy - hbase;
        float gx1 = cx + hbase;
        float gy1 = cy + hbase;

        if( cy < batch_height && cx < batch_width )
        {
            int gt_index = -1;
            float max_intersection = 0.0;

            for( int i=0 ; i<nrois ; i++ )
            {
                const float* roi = gtboxes + i*4;
                int roi_batch = gtbatches[i];

                if( roi_batch == b )
                {
                    // Calculating the intersection between the grid cell and
                    // the bounding box

                    float ix = min( roi[2], gx1 ) - max( roi[0], gx0 );
                    float iy = min( roi[3], gy1 ) - max( roi[1], gy0 );
                    float intersection = max(0.0,static_cast<double>(ix)+1)*max(0.0,static_cast<double>(iy)+1);

                    if( intersection > max_intersection )
                    {
                        gt_index = i;
                        max_intersection = intersection;
                    }
                }
            }

            if( gt_index == -1 )
            {
                labels[ index ] = 0;
            }
            else
            {
                // In our labeling, background is 0 and objects are counted from 1
                labels[ index ] = static_cast<int>( gtlabels[ gt_index ] );

                // Locating the target pointer
                float* target = targets + index * 4;

                // Locating the ROI pointer
                const float* roi = gtboxes + gt_index*4;

                // Encoding the bounding box
                float roi_x = ( roi[2] + roi[0] ) * 0.5;
                float roi_y = ( roi[3] + roi[1] ) * 0.5;
                float roi_w = roi[2] - roi[0];
                float roi_h = roi[3] - roi[1];

                target[0] = ( roi_x - cx ) / roi_w;
                target[1] = ( roi_y - cy ) / roi_h;
                target[2] = log( roi_w );
                target[3] = log( roi_h );
            }

        }
        else
        {
            labels[ index ] = -1;
        }
    }

}

void proposal_encode_fill( float* ptr, int total, float value )
{
    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    ProposalEncodeFill<<<total_blocks,kThreadsPerBlock>>>( total, ptr, value );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}


void proposal_encode_forward_gpu( const float* gtboxes, const float* gtlabels, const int* gtbatches,
                            const float* shapes, int nrois, float spatial_scale,
                            float* labels, float* targets, int nbatches, int height, int width )
{
    int total = nbatches*height*width*4;
    proposal_encode_fill( targets, total, 0.0 );

    total = nbatches*height*width;

    cudaError_t err;
    int total_blocks = (total+kThreadsPerBlock-1)/kThreadsPerBlock;

    ProposalEncodeForward<<<total_blocks,kThreadsPerBlock>>>( total, gtboxes, gtlabels, gtbatches, shapes, nrois,
                                                              spatial_scale, labels, targets,
                                                              nbatches, height, width );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#endif
