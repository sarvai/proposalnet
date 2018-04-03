#include <iostream>
#include <cmath>
#include <cfloat>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "frcnn_prepare_op_gpu.hpp"
// Should use GPU
#else
#define DEVICE_ DEVICE_CPU
#include "frcnn_prepare_op_cpu.hpp"
// Should use CPU
#endif

using namespace tensorflow;
using namespace std;

REGISTER_OP("FrcnnPrepare")
.Attr("nclasses : int")
.Attr("fg_min_overlap : float")
.Attr("bg_max_overlap : float")
.Input("gtboxes : float")
.Input("gtlabels : float ")
.Input("gtbatches : int32")
.Input("rois : float")
.Input("roilabels : float")
.Input("roibatches : int32")
.Input("overlaps : float")
.Output("labels : float")
.Output("deltas : float")
;

class FrcnnPrepareOp : public OpKernel
{
private:
    int nclasses;
    float fg_min_overlap;
    float bg_max_overlap;
public:
    explicit FrcnnPrepareOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("nclasses",&nclasses) );
        OP_REQUIRES_OK( construct, construct->GetAttr("fg_min_overlap",&fg_min_overlap) );
        OP_REQUIRES_OK( construct, construct->GetAttr("bg_max_overlap",&bg_max_overlap) );
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

        const Tensor& overlaps_tensor = context -> input(6);
        OP_REQUIRES( context, overlaps_tensor.dims() == 2,
                        errors::InvalidArgument("overlaps should have 2 dimensions") );
        OP_REQUIRES( context, overlaps_tensor.dim_size(0) == nrois,
                        errors::InvalidArgument("overlaps and rois must have the same number of rows") );
        OP_REQUIRES( context, overlaps_tensor.dim_size(1) == ngtboxes,
                        errors::InvalidArgument("Each row of overlaps must have ngtboxes dimensions") );
        auto overlaps = overlaps_tensor.flat<float>();

        int labels_dims[2];
        labels_dims[0] = nrois;
        labels_dims[1] = nclasses;

        int deltas_dims[2];
        deltas_dims[0] = nrois;
        deltas_dims[1] = nclasses*4;

        TensorShape labels_shape;
        auto t0 = TensorShapeUtils::MakeShape(labels_dims, 2, &labels_shape);

        TensorShape deltas_shape;
        auto t1 = TensorShapeUtils::MakeShape(deltas_dims, 2, &deltas_shape);

        Tensor* labels_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(0, labels_shape, &labels_tensor) );
        auto labels = labels_tensor -> flat<float>();

        Tensor* deltas_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(1, deltas_shape, &deltas_tensor ) );
        auto deltas = deltas_tensor -> flat<float>();

#ifdef GOOGLE_CUDA
        frcnn_prepare_forward_gpu( gtboxes.data(), gtlabels.data(), gtbatches.data(), ngtboxes,
                                   rois.data(), roilabels.data(), roibatches.data(), nrois,
                                   overlaps.data(),
                                   nclasses, fg_min_overlap, bg_max_overlap,
                                   labels.data(), deltas.data() );
#else
        frcnn_prepare_forward_cpu( gtboxes.data(), gtlabels.data(), gtbatches.data(), ngtboxes,
                                   rois.data(), roilabels.data(), roibatches.data(), nrois,
                                   overlaps.data(),
                                   nclasses, fg_min_overlap, bg_max_overlap,
                                   labels.data(), deltas.data() );
#endif

    }
};

REGISTER_KERNEL_BUILDER(Name("FrcnnPrepare").Device(DEVICE_), FrcnnPrepareOp);
