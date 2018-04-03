#include <cmath>
#include <cfloat>
#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "roi_pool_op_gpu.hpp"
#else
#define DEVICE_ DEVICE_CPU
#include "roi_pool_op_cpu.hpp"
#endif

using namespace tensorflow;

REGISTER_OP("RoiPool")
    .Attr("pooled_height : int")
    .Attr("pooled_width : int")
    .Attr("spatial_scale : float")
    .Input("input : float")
    .Input("rois : float")
    .Output("output : float")
    .Output("argmax : int32")
    ;

REGISTER_OP("RoiPoolGrad")
    .Input("input : float")
    .Input("argmax : int32")
    .Input("grad : float")
    .Output("input_grad : float")
;

class RoiPoolOp : public OpKernel
{
public:
    explicit RoiPoolOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("pooled_height",&pooled_height) );
        OP_REQUIRES(construct, pooled_height >= 0, errors::InvalidArgument("Need pooled_height >= 0, got ", pooled_height));
        OP_REQUIRES_OK( construct, construct->GetAttr("pooled_width",&pooled_width) );
        OP_REQUIRES(construct, pooled_width >= 0, errors::InvalidArgument("Need pooled_width >= 0, got ", pooled_width));
        OP_REQUIRES_OK( construct, construct->GetAttr("spatial_scale",&spatial_scale) );
    }
    void Compute( OpKernelContext* context ) override
    {
        const Tensor& input_tensor = context -> input(0);
        const Tensor& rois_tensor = context -> input(1);

        OP_REQUIRES( context, input_tensor.dims() == 4, errors::InvalidArgument("Input must be 4 dimensional") );
        OP_REQUIRES( context, rois_tensor.dims() == 2, errors::InvalidArgument("ROIs must be 2 dimensional") );

        auto input = input_tensor.flat<float>();
        auto rois = rois_tensor.flat<float>();

        int nrois = rois_tensor.dim_size(0);
        int batch_size = input_tensor.dim_size(0);
        int input_height = input_tensor.dim_size(1);
        int input_width = input_tensor.dim_size(2);
        int nchannels = input_tensor.dim_size(3);

        int dims[4];
        dims[0] = nrois;
        dims[1] = pooled_height;
        dims[2] = pooled_width;
        dims[3] = nchannels;

        TensorShape output_shape;
        auto t0 = TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor -> flat<float>();

        Tensor* argmax_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
        auto argmax = argmax_tensor -> flat<int32>();

        #ifdef GOOGLE_CUDA
        roi_pool_forward_gpu(
            input.data(), input_height, input_width, nchannels, pooled_height, pooled_width, spatial_scale,
            rois.data(), nrois, output.data(), argmax.data()
        );
        #else
        roi_pool_forward_cpu(
            input.data(), input_height, input_width, nchannels, pooled_height, pooled_width, spatial_scale,
            rois.data(), nrois, output.data(), argmax.data()
        );
        #endif
    }
private:
    int pooled_height;
    int pooled_width;
    float spatial_scale;
};

class RoiPoolGradOp : public OpKernel
{
public:
    explicit RoiPoolGradOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {}

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& input_tensor = context -> input(0);
        const Tensor& argmax_tensor = context -> input(1);
        const Tensor& grad_tensor = context -> input(2);

        TensorShape input_grad_shape = input_tensor.shape();
        Tensor* input_grad_tensor = NULL;
        OP_REQUIRES_OK(context, context -> allocate_output(0, input_grad_shape, &input_grad_tensor));

        auto argmax = argmax_tensor.flat<int32>();
        auto top_grad = grad_tensor.flat<float>();
        auto bottom_grad = input_grad_tensor -> flat<float>();

        #ifdef GOOGLE_CUDA
        roi_pool_backward_gpu( argmax.data(), top_grad.data(), top_grad.size(), bottom_grad.data(), bottom_grad.size() );
        #else
        roi_pool_backward_cpu( argmax.data(), top_grad.data(), top_grad.size(), bottom_grad.data(), bottom_grad.size() );
        #endif
    }
private:
};

REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_), RoiPoolOp);
REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_), RoiPoolGradOp);
