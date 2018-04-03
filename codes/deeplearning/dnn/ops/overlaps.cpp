#include <iostream>
#include <cmath>
#include <cfloat>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "overlaps_op_gpu.hpp"
// Should use GPU
#else
#define DEVICE_ DEVICE_CPU
#include "overlaps_op_cpu.hpp"
// Should use CPU
#endif

using namespace tensorflow;
using namespace std;

REGISTER_OP("Overlaps")
.Input("gtboxes : float")
.Input("gtlabels : float ")
.Input("gtbatches : int32")
.Input("rois : float")
.Input("roilabels : float")
.Input("roibatches : int32")
.Output("overlaps : float")
;

class OverlapsOp : public OpKernel
{
private:
public:
    explicit OverlapsOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
    }

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& gtboxes_tensor = context -> input(0);
        OP_REQUIRES( context, gtboxes_tensor.dims() == 2,
                        errors::InvalidArgument("gtboxes should have 2 dimensions") );
        OP_REQUIRES( context, gtboxes_tensor.dim_size(1) == 4,
                        errors::InvalidArgument("Each row of gtboxes must have 4 dimensions") );
        int ngtboxes = gtboxes_tensor.dim_size(0);
        auto gtboxes = gtboxes_tensor.flat<float>();

        const Tensor& gtlabels_tensor = context -> input(1);
        OP_REQUIRES( context, gtlabels_tensor.dims() == 2,
                        errors::InvalidArgument("gtlabels should have 2 dimensions") );
        OP_REQUIRES( context, gtlabels_tensor.dim_size(0) == ngtboxes,
                        errors::InvalidArgument("gtlabels and gtboxes must have the same number of rows") );
        OP_REQUIRES( context, gtlabels_tensor.dim_size(1) == 1,
                        errors::InvalidArgument("Each row of gtlabels must have 1 dimensions") );
        auto gtlabels = gtlabels_tensor.flat<float>();

        const Tensor& gtbatches_tensor = context -> input(2);
        OP_REQUIRES( context, gtbatches_tensor.dims() == 2,
                        errors::InvalidArgument("gtbatches should have 2 dimensions") );
        OP_REQUIRES( context, gtbatches_tensor.dim_size(0) == ngtboxes,
                        errors::InvalidArgument("gtbatches and gtboxes must have the same number of rows") );
        OP_REQUIRES( context, gtbatches_tensor.dim_size(1) == 1,
                        errors::InvalidArgument("Each row of gtbatches must have 1 dimensions") );
        auto gtbatches = gtbatches_tensor.flat<int>();

        const Tensor& rois_tensor = context -> input(3);
        OP_REQUIRES( context, rois_tensor.dims() == 2,
                        errors::InvalidArgument("rois should have 2 dimensions") );
        OP_REQUIRES( context, rois_tensor.dim_size(1) == 4,
                        errors::InvalidArgument("Each row of rois must have 4 dimensions") );
        int nrois = rois_tensor.dim_size(0);
        auto rois = rois_tensor.flat<float>();

        const Tensor& roilabels_tensor = context -> input(4);
        OP_REQUIRES( context, roilabels_tensor.dims() == 2,
                        errors::InvalidArgument("roilabels should have 2 dimensions") );
        OP_REQUIRES( context, roilabels_tensor.dim_size(0) == nrois,
                        errors::InvalidArgument("roilabels and rois must have the same number of rows") );
        OP_REQUIRES( context, roilabels_tensor.dim_size(1) == 1,
                        errors::InvalidArgument("Each row of roilabels must have 1 dimensions") );
        auto roilabels = roilabels_tensor.flat<float>();

        const Tensor& roibatches_tensor = context -> input(5);
        OP_REQUIRES( context, roibatches_tensor.dims() == 2,
                        errors::InvalidArgument("roibatches should have 2 dimensions") );
        OP_REQUIRES( context, roibatches_tensor.dim_size(0) == nrois,
                        errors::InvalidArgument("roibatches and rois must have the same number of rows") );
        OP_REQUIRES( context, roibatches_tensor.dim_size(1) == 1,
                        errors::InvalidArgument("Each row of roibatches must have 1 dimensions") );
        auto roibatches = roibatches_tensor.flat<int>();

        int overlaps_dims[2];
        overlaps_dims[0] = nrois;
        overlaps_dims[1] = ngtboxes;

        TensorShape overlaps_shape;
        auto t0 = TensorShapeUtils::MakeShape(overlaps_dims, 2, &overlaps_shape);

        Tensor* overlaps_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(0, overlaps_shape, &overlaps_tensor) );
        auto overlaps = overlaps_tensor -> flat<float>();


#ifdef GOOGLE_CUDA
        overlaps_forward_gpu( gtboxes.data(), gtlabels.data(), gtbatches.data(), ngtboxes,
                                   rois.data(), roilabels.data(), roibatches.data(), nrois,
                                   overlaps.data() );
#else
        overlaps_forward_cpu( gtboxes.data(), gtlabels.data(), gtbatches.data(), ngtboxes,
                                   rois.data(), roilabels.data(), roibatches.data(), nrois,
                                   overlaps.data() );
#endif

    }
};

REGISTER_KERNEL_BUILDER(Name("Overlaps").Device(DEVICE_), OverlapsOp);
