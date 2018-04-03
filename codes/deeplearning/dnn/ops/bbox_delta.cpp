#include <iostream>
#include <cmath>
#include <cfloat>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "bbox_delta_gpu.hpp"
//#include "proposal_decode_op_gpu.hpp"
// Should use GPU
#else
#define DEVICE_ DEVICE_CPU
#include "bbox_delta_cpu.hpp"
// Should use CPU
#endif

using namespace tensorflow;

REGISTER_OP("BboxDeltaInv")
.Input("rois : float")
.Input("deltas : float")
.Output("output : float")
;

class BboxDeltaInvOp : public OpKernel
{
private:
public:
    explicit BboxDeltaInvOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {}

    void Compute( OpKernelContext* context ) override
    {
        // Processing Input
        const Tensor& rois_tensor = context -> input(0);
        OP_REQUIRES( context, rois_tensor.dims() == 2, errors::InvalidArgument("Rois must be 2 dimensional") );
        OP_REQUIRES( context, rois_tensor.dim_size(1) == 4, errors::InvalidArgument("Each row of Rois must be 4 elements"));
        const Tensor& deltas_tensor = context -> input(1);
        OP_REQUIRES( context, deltas_tensor.dims() == 2, errors::InvalidArgument("Deltas must be 2 dimensional") );
        OP_REQUIRES( context, deltas_tensor.dim_size(1) == 4, errors::InvalidArgument("Each row of Deltas must be 4 elements"));
        OP_REQUIRES( context, rois_tensor.dim_size(0) == deltas_tensor.dim_size(0), errors::InvalidArgument("Rois and Deltas must have the same number of rows") );

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context -> allocate_output(0, rois_tensor.shape(), &output_tensor ));

        int count = rois_tensor.dim_size(0);
        auto rois = rois_tensor.flat<float>();
        auto deltas = deltas_tensor.flat<float>();
        auto output = output_tensor -> flat<float>();

        #ifdef GOOGLE_CUDA
        bbox_delta_inv_forward_gpu( rois.data(), deltas.data(), count, output.data() );
        #else
        bbox_delta_inv_forward_cpu( rois.data(), deltas.data(), count, output.data() );
        #endif


    }
};

REGISTER_KERNEL_BUILDER(Name("BboxDeltaInv").Device(DEVICE_), BboxDeltaInvOp);
