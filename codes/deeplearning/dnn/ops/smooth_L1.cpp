#include <iostream>
#include <cmath>
#include <cfloat>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef GOOGLE_CUDA
#define DEVICE_ DEVICE_GPU
#include "smooth_L1_gpu.hpp"
#else
#define DEVICE_ DEVICE_CPU
#include "smooth_L1_cpu.hpp"
#endif

using namespace tensorflow;

REGISTER_OP("SmoothL1")
    .Attr("sigma : float")
	.Input("diffs : float")
	.Output("output : float")
;

REGISTER_OP("SmoothL1Grad")
    .Attr("sigma : float")
    .Input("diffs : float")
    .Input("top_grad : float")
    .Output("bottom_grad : float")
;

class SmoothL1Op : public OpKernel
{
private:
    float sigma;
    float sigma2;
public:
    explicit SmoothL1Op( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("sigma",&sigma) );
        OP_REQUIRES(construct, sigma > 0, errors::InvalidArgument("sigma needs to be > 0, got ", sigma));
        sigma2 = sigma*sigma;
    }

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& diffs_tensor = context -> input(0);

        OP_REQUIRES( context, diffs_tensor.dims() == 2, errors::InvalidArgument("preds must be 2 dimensional") );

        int batch_size = diffs_tensor.dim_size(0);
        int dim = diffs_tensor.dim_size(1);

        Tensor* outputs_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, diffs_tensor.shape(), &outputs_tensor));

        int ndata = batch_size * dim;

        auto diffs = diffs_tensor.flat<float>();
        auto outputs = outputs_tensor -> flat<float>();

        #ifdef GOOGLE_CUDA
        smooth_l1_forward_gpu( diffs.data(), outputs.data(), sigma2, ndata );
        #else
        smooth_l1_forward_cpu( diffs.data(), outputs.data(), sigma2, ndata );
        #endif
    }

};

class SmoothL1GradOp : public OpKernel
{
private:
    float sigma;
    float sigma2;
public:
    explicit SmoothL1GradOp( OpKernelConstruction* construct ) :  OpKernel(construct)
    {
        OP_REQUIRES_OK( construct, construct->GetAttr("sigma",&sigma) );
        OP_REQUIRES(construct, sigma > 0, errors::InvalidArgument("sigma needs to be > 0, got ", sigma));
        sigma2 = sigma*sigma;
    }

    void Compute( OpKernelContext* context ) override
    {
        const Tensor& diffs_tensor = context -> input(0);
        const Tensor& top_grad_tensor = context -> input(1);

        OP_REQUIRES( context, diffs_tensor.dims() == 2, errors::InvalidArgument("preds must be 2 dimensional") );
        OP_REQUIRES( context, top_grad_tensor.dims() == 2, errors::InvalidArgument("top grad must be 2 dimensional") );

        OP_REQUIRES( context, top_grad_tensor.dim_size(0) == diffs_tensor.dim_size(0), errors::InvalidArgument("Top grad dimension 0 mismatch") );
        OP_REQUIRES( context, top_grad_tensor.dim_size(1) == diffs_tensor.dim_size(1), errors::InvalidArgument("Top grad dimension 1 mismatch") );

        Tensor* bottom_grad_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, diffs_tensor.shape(), &bottom_grad_tensor));

        int batch_size = diffs_tensor.dim_size(0);
        int dim = diffs_tensor.dim_size(1);

        int ndata = batch_size * dim;
        auto diffs = diffs_tensor.flat<float>();
        auto top_grad = top_grad_tensor.flat<float>();
        auto bottom_grad = bottom_grad_tensor -> flat<float>();

        #ifdef GOOGLE_CUDA
        smooth_l1_backward_gpu( diffs.data(), top_grad.data(), bottom_grad.data(), sigma2, ndata );
        #else
        smooth_l1_backward_cpu( diffs.data(), top_grad.data(), bottom_grad.data(), sigma2, ndata );
        #endif
    }
};

REGISTER_KERNEL_BUILDER(Name("SmoothL1").Device(DEVICE_), SmoothL1Op);
REGISTER_KERNEL_BUILDER(Name("SmoothL1Grad").Device(DEVICE_), SmoothL1GradOp);
