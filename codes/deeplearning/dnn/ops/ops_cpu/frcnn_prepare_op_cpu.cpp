#include "frcnn_prepare_op_cpu.hpp"

void fill( int total, float* arr, float value )
{
    for( int i=0 ; i<total ; i++ )
        arr[i] = value;
}

void frcnn_prepare_forward_cpu( const float* gtboxes, const float* gtlabels, const int* gtbatches, int ngtboxes,
                               const float* rois, const float* roilabels, const int* roibatches, int nrois,
                               const float* overlaps,
                               int nclasses, float fg_min_overlap, float bg_max_overlap,
                               float* labels, float* deltas )
{
    fill( nrois*nclasses, labels, -1.0 );
    fill( nrois*nclasses*4, deltas, 0.0 );

    for( int index=0 ; index<nrois; index++ )
    {
        const float* roi_overlaps = overlaps + index*ngtboxes;

        int argmax = -1;
        float max = -10;

        for( int i=0 ; i<ngtboxes ; i++ )
        {
            if( roi_overlaps[i] > max )
            {
                argmax = i;
                max = roi_overlaps[i];
            }
        }

        if( max > -1 && argmax > -1 )
        {
            // -1 is due to the fact that category indices start from 1
            int cls = gtlabels[ argmax ] - 1;

            // The label remains -1 if the overlap is between min_fg and max_bg overlaps

            if( max >= fg_min_overlap )
            {
                // This is positive
                labels[ index * nclasses + cls ] = 1.0;

                const float* box = gtboxes + argmax*4;
                const float* roi = rois + index*4;
                float* delta = deltas + index * (nclasses*4) + cls*4;

                // Box is the target
                float box_w = box[2] - box[0];
                float box_h = box[3] - box[1];
                float box_x = box[0] + box_w*0.5;
                float box_y = box[1] + box_h*0.5;

                // Roi is the reference
                float roi_w = roi[2] - roi[0];
                float roi_h = roi[3] - roi[1];
                float roi_x = roi[0] + roi_w*0.5;
                float roi_y = roi[1] + roi_h*0.5;

                delta[0] = ( box_x - roi_x ) / roi_w; // dx
                delta[1] = ( box_y - roi_y ) / roi_h; // dy
                delta[2] = log( box_w / roi_w ); // dw
                delta[3] = log( box_h / roi_h ); // dh
            }
            else if( max < bg_max_overlap )
            {
                // This is negative
                labels[ index * nclasses + cls ] = 0.0;
            }
        }
    }
}
