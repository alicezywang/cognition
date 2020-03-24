#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <functional>

#include <sys/time.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

using namespace tensorflow;

REGISTER_OP("Time")
    .Input("input_data: float")
    .Output("output_data: float")
    .Output("time: double")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);
class TimeOp : public OpKernel {
public:
  explicit TimeOp(OpKernelConstruction* context) : OpKernel(context)
  {
  }
  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double timestamp = (double)tv.tv_sec*1000.0 + (double)tv.tv_usec/1000.0;
    //std::cout << "---------- " << std::fixed << timestamp << " ---------- " << this->name() << "::"<<this->input_type(0) <<std::endl;
    const Tensor &input_data = context->input(0);
    context->set_output(0, input_data);
    Tensor *output_data = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(), &output_data));
    output_data->scalar<double>()(0) = (double)timestamp;
  }
private:
};

Status TimeGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(grad_inputs[0]);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_KERNEL_BUILDER(Name("Time").Device(DEVICE_CPU), TimeOp);
REGISTER_GRADIENT_OP("Time", TimeGrad);
