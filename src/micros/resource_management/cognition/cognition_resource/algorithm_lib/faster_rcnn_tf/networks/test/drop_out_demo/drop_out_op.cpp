#include <cstdlib>
#include <ctime>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("DropOut").Input("to_drop: float").Output("output: float").SetShapeFn(tensorflow::shape_inference::UnchangedShape);

class DropOutOp : public OpKernel
{
public:
    DropOutOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext *context)
    {
        int i; const float keep_prob = 0.5f;
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->flat<float>();
        const int n = input.size();
        std::srand(std::time(0));
        for(i=0; i<n; i++)
        {
            output(i) = ((std::rand() & 65535) < (int)((double)keep_prob * 65536.0)) ? input(i) / keep_prob : 0.0f;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("DropOut").Device(DEVICE_CPU), DropOutOp);
