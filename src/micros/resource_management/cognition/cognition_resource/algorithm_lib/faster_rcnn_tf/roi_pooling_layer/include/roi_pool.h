// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_USER_OPS_ROI_POOL_H_
#define TENSORFLOW_USER_OPS_ROI_POOL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup user_ops User Ops
/// @{

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` top_data
/// * `Output` argmax
class RoiPool {
 public:
  RoiPool(const ::tensorflow::Scope& scope, ::tensorflow::Input bottom_data,
        ::tensorflow::Input bottom_rois, int64 pooled_height, int64
        pooled_width, float spatial_scale);

  ::tensorflow::Output top_data;
  ::tensorflow::Output argmax;
};

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class RoiPoolGrad {
 public:
  RoiPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input bottom_data,
            ::tensorflow::Input bottom_rois, ::tensorflow::Input argmax,
            ::tensorflow::Input grad, int64 pooled_height, int64 pooled_width,
            float spatial_scale);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_USER_OPS_H_
