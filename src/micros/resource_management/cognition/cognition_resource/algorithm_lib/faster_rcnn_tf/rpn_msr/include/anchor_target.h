#ifndef RPN_MSR_INCLUDE_ANCHOR_TARGET_H
#define RPN_MSR_INCLUDE_ANCHOR_TARGET_H

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow
{
    namespace ops
    {
        class AnchorTarget
        {
        public:
            AnchorTarget(const ::tensorflow::Scope& scope, ::tensorflow::Input rpn_cls_score,
                         ::tensorflow::Input gt_boxes, ::tensorflow::Input im_info,
                         ::tensorflow::Input data, int64 feat_stride);
            ::tensorflow::Output rpn_labels;
            ::tensorflow::Output rpn_bbox_targets;
            ::tensorflow::Output rpn_bbox_inside_weights;
            ::tensorflow::Output rpn_bbox_outside_weights;
        };
    }
}

#endif
