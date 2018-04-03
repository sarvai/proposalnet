#include "proposal_encode_op_cpu.hpp"

void fill( float* ptr, int count, float value )
{
    for( int i=0 ; i<count ; i++ )
        ptr[i] = value;
}

void proposal_encode_forward_cpu( const float* gtboxes, const float* gtlabels, const int* gtbatches,
                            const float* shapes, int nrois, float spatial_scale,
                            float* labels, float* targets, int nbatches, int height, int width )
{
    //cout << nrois << "\t" << nclasses << "\t" << spa:tial_scale << endl;
    //cout << nbatches << "\t" << height << "\t" << width << endl;

    int base = round( 1.0/spatial_scale );
    int total = nbatches * height * width;
    float hbase = static_cast<float>( base ) / 2.0; // Half base

    fill( targets, total * 4, 0.0 );

    for( int index=0 ; index<total ; index++ )
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
                // Invalid regions are labeled as -1
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
