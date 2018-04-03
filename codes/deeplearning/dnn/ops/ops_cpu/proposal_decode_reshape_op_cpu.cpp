#include "proposal_decode_reshape_op_cpu.hpp"

void proposal_decode_reshape_forward_cpu( const float* input, int nbatches, int height, int width, int nchannels,
                double spatial_scale, float* output )
{
    int base = round( 1.0 / spatial_scale );
    int total = nbatches * height * width;

    float hbase = static_cast<float>(base)/2;

    for( int index=0 ; index<total ; index++ )
    {
        int grid_w = index % width;
        int grid_h = (index / width) % height;
        int grid_n = (index / (height*width));

        float px = grid_w * base + hbase;
        float py = grid_h * base + hbase;

        float crx = input[ index*4 ];
        float cry = input[ index*4+1 ];
        float crlogw = input[ index*4+2 ];
        float crlogh = input[ index*4+3 ];

        float w = exp( crlogw );
        float h = exp( crlogh );
        float cx = w * crx + px;
        float cy = h * cry + py;

        output[ index*5 ] = grid_n;
        output[ index*5+1 ] = cx - w*0.5;
        output[ index*5+2 ] = cy - h*0.5;
        output[ index*5+3 ] = cx + w*0.5;
        output[ index*5+4 ] = cy + h*0.5;
    }

}
