#include "roi_pool_op_cpu.hpp"

void roi_pool_forward_cpu(
    const float* input, const int height, const int width, const int nchannels, const int pooled_height,
    const int pooled_width, const float spatial_scale, const float* rois, const int nrois, float* output,
    int* argmax
)
{
    int noutput = nrois*pooled_height*pooled_width*nchannels;

    int input_base_hwc = height*width*nchannels;
    int input_base_wc = width*nchannels;
    int input_base_c = nchannels;

    int output_base_hwc = pooled_height*pooled_width*nchannels;
    int output_base_wc = pooled_width*nchannels;
    int output_base_c = nchannels;

    for( int i=0 ; i<noutput; i++ )
    {
        int c = i % output_base_c;
        int pw = (i / (output_base_c)) % pooled_width;
        int ph = (i / (output_base_wc)) % pooled_height;
        int n = (i / (output_base_hwc));

        const float* roi = rois + n * 5;

        int batch_index = static_cast<int>( roi[0] );
        int roi_x0 = static_cast<int>( round(roi[1]*spatial_scale) );
        int roi_y0 = static_cast<int>( round(roi[2]*spatial_scale) );
        int roi_x1 = static_cast<int>( round(roi[3]*spatial_scale) );
        int roi_y1 = static_cast<int>( round(roi[4]*spatial_scale) );

        int roi_width = std::max( roi_x1 - roi_x0 + 1, 1 );
        int roi_height = std::max( roi_y1 - roi_y0 + 1, 1 );

        float bin_size_h = static_cast<float>( roi_height ) / static_cast<float>( pooled_height );
        float bin_size_w = static_cast<float>( roi_width ) / static_cast<float>( pooled_width );

        int hstart = roi_y0 + static_cast<int>( floor( ph * bin_size_h ) );
        hstart = std::min( std::max( hstart, 0 ), height );

        int hend = roi_y0 + static_cast<int>( ceil( (ph+1) * bin_size_h ) );
        hend = std::min( std::max( hend, 0 ), height );

        int wstart = roi_x0 + static_cast<int>(  floor( pw * bin_size_w ) );
        wstart = std::min( std::max( wstart, 0 ), width );

        int wend = roi_x0 + static_cast<int>( ceil( (pw+1) * bin_size_w ) );
        wend = std::min( std::max( wend, 0 ), width );

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

        output[i] = maxval;
        argmax[i] = maxidx;
    }
}

void roi_pool_backward_cpu(
    const int* argmax, const float* top_grad, const int top_count, float* bottom_grad, const int bottom_count
)
{
    for( int i=0 ; i<bottom_count ; i++ )
        bottom_grad[i] = 0.0;

    for( int i=0 ; i<top_count ; i++ )
    {
        int idx = argmax[i];
        bottom_grad[ idx ] += top_grad[i];
    }
}
