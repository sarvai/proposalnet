#include <iostream>
#include <cmath>
#include <cfloat>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "proposal_decode_reshape_op_gpu.hpp"
// Should use GPU
#else
#define DEVICE_ DEVICE_CPU
#include "proposal_decode_reshape_op_cpu.hpp"
// Should use CPU
#endif

using namespace tensorflow;

REGISTER_OP("ProposalDecodeReshape")
.Attr("spatial_scale : float")
.Input("bottom : float")
.Output("top : float")
;

class ProposalDecodeReshapeOp : public OpKernel
{
public:
    explicit ProposalDecodeReshapeOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("spatial_scale",&spatial_scale) );
    }

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& input_tensor = context -> input(0);
        OP_REQUIRES( context, input_tensor.dims() == 4, errors::InvalidArgument("Input must be 4 dimensional") );

        int nbatches = input_tensor.dim_size(0);
        int height = input_tensor.dim_size(1);
        int width = input_tensor.dim_size(2);
        int nchannels = input_tensor.dim_size(3);

        OP_REQUIRES( context, nchannels == 4, errors::InvalidArgument("Number of channels must be 4") );

        int dims[2];
        dims[0] = nbatches*height*width;
        dims[1] = 5;

        TensorShape output_shape;
        auto t0 = TensorShapeUtils::MakeShape(dims, 2, &output_shape);


        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(0, output_shape, &output_tensor ));

        auto input = input_tensor.flat<float>();
        auto output = output_tensor -> flat<float>();

        #ifdef GOOGLE_CUDA
        proposal_decode_reshape_forward_gpu( input.data(), nbatches, height, width, nchannels, spatial_scale, output.data() );
        #else
        proposal_decode_reshape_forward_cpu( input.data(), nbatches, height, width, nchannels, spatial_scale, output.data() );
        #endif
    }
private:
    float spatial_scale;
};

REGISTER_KERNEL_BUILDER(Name("ProposalDecodeReshape").Device(DEVICE_), ProposalDecodeReshapeOp);
