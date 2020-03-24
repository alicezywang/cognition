#ifndef RPN_MSR_INCLUDE_PROPOSAL_TARGET_H
#define RPN_MSR_INCLUDE_PROPOSAL_TARGET_H

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

class ProposalTarget {
 public:
  ProposalTarget(const ::tensorflow::Scope& scope, ::tensorflow::Input rpn_rois,
               ::tensorflow::Input gt_boxes, int64 num_classes);

  ::tensorflow::Output rois;
  ::tensorflow::Output labels;
  ::tensorflow::Output bbox_targets;
  ::tensorflow::Output bbox_inside_weights;
  ::tensorflow::Output bbox_outside_weights;
};

}
}

#endif
