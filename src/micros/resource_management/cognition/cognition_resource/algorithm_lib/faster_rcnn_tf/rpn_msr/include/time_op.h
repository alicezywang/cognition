#ifndef RPN_MSR_INCLUDE_TIME_OP_H
#define RPN_MSR_INCLUDE_TIME_OP_H

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

class Time {
 public:
  Time(const ::tensorflow::Scope& scope, ::tensorflow::Input input_data);

  ::tensorflow::Output output_data;
  ::tensorflow::Output time;
};

}
}

#endif
