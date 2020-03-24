#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut").Input("to_zero: float").Output("output: float").SetShapeFn(tensorflow::shape_inference::UnknownShape);

class ZeroOutOp : public OpKernel
{
public:
    ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext *context)
    {
        int i;
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<float>();
        const int n = input.size();
        for(i=1; i<n; i++)
        {
            output(i) = 0.0f;
        }
        if (n>0) output(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
