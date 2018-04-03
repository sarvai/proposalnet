#include <iostream>
#include <cmath>
#include <cfloat>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "proposal_encode_op_gpu.hpp"
// Should use GPU
#else
#define DEVICE_ DEVICE_CPU
#include "proposal_encode_op_cpu.hpp"
// Should use CPU
#endif

using namespace tensorflow;
using namespace std;

REGISTER_OP("ProposalEncode")
.Attr("spatial_scale : float")
.Input("feat_map : float")
.Input("gtboxes : float ")
.Input("gtlabels : float")
.Input("gtbatches : int32")
.Input("shapes : float")
.Output("labels : float")
.Output("targets : float")
;

class ProposalEncodeOp : public OpKernel
{
private:
    float spatial_scale;
public:
    explicit ProposalEncodeOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("spatial_scale",&spatial_scale) );
    }

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& feat_map_tensor = context -> input(0);
        OP_REQUIRES( context, feat_map_tensor.dims() == 4,
                errors::InvalidArgument("Feature Map must be 4 dimensional") );

        int nbatches = feat_map_tensor.dim_size(0);
        int height = feat_map_tensor.dim_size(1);
        int width = feat_map_tensor.dim_size(2);
        int nchannels = feat_map_tensor.dim_size(3);

        const Tensor& gtboxes_tensor = context -> input(1);
        const Tensor& gtlabels_tensor = context -> input(2);
        const Tensor& gtbatches_tensor = context -> input(3);
        const Tensor& shapes_tensor = context -> input(4);

        int nrois = gtboxes_tensor.dim_size(0);

        OP_REQUIRES( context, gtboxes_tensor.dim_size(0) == nrois,
                errors::InvalidArgument("Number of gtlabels should be equal to gtboxes") );

        OP_REQUIRES( context, gtlabels_tensor.dims() == 2,
                errors::InvalidArgument("gtlabels must be 2 dimensional") );

        OP_REQUIRES( context, gtlabels_tensor.dim_size(1) == 1,
                errors::InvalidArgument("gtlabels must have 1 columns") );

        OP_REQUIRES( context, gtbatches_tensor.dim_size(0) == nrois,
                errors::InvalidArgument("Number of gtbatches should be equal to gtboxes") );

        OP_REQUIRES( context, gtbatches_tensor.dims() == 2,
                errors::InvalidArgument("gtbatches must be 2 dimensional") );

        OP_REQUIRES( context, gtbatches_tensor.dim_size(1) == 1,
                errors::InvalidArgument("gtbatches should have only one column") );

        OP_REQUIRES( context, shapes_tensor.dims() == 2,
                errors::InvalidArgument("shapes must be 2 dimensional") );

        auto gtboxes = gtboxes_tensor.flat<float>();
        auto gtlabels = gtlabels_tensor.flat<float>();
        auto gtbatches = gtbatches_tensor.flat<int>();
        auto shapes = shapes_tensor.flat<float>();

        int labels_dims[4];
        labels_dims[0] = nbatches;
        labels_dims[1] = height;
        labels_dims[2] = width;
        labels_dims[3] = 1;

        TensorShape labels_shape;
        auto t0 = TensorShapeUtils::MakeShape(labels_dims, 4, &labels_shape);

        int targets_dims[4];
        targets_dims[0] = nbatches;
        targets_dims[1] = height;
        targets_dims[2] = width;
        targets_dims[3] = 4;

        TensorShape targets_shape;
        auto t1 = TensorShapeUtils::MakeShape(targets_dims, 4, &targets_shape);

        // Class labeling of the proposals
        Tensor* labels_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(0, labels_shape, &labels_tensor ));
        auto labels = labels_tensor -> flat<float>();

        // Bounding box targets of the proposals
        // Must be set to zeros
        Tensor* targets_tensor = NULL;
        OP_REQUIRES_OK( context, context -> allocate_output(1, targets_shape, &targets_tensor ));
        auto targets = targets_tensor -> flat<float>();

#ifdef GOOGLE_CUDA
        proposal_encode_forward_gpu( gtboxes.data(), gtlabels.data(), gtbatches.data(),
                                        shapes.data(), nrois, spatial_scale,
                                        labels.data(), targets.data(), nbatches, height,
                                        width );

#else
        proposal_encode_forward_cpu( gtboxes.data(), gtlabels.data(), gtbatches.data(),
                                        shapes.data(), nrois, spatial_scale,
                                        labels.data(), targets.data(), nbatches, height,
                                        width );
#endif
    }

};

REGISTER_KERNEL_BUILDER(Name("ProposalEncode").Device(DEVICE_), ProposalEncodeOp);
