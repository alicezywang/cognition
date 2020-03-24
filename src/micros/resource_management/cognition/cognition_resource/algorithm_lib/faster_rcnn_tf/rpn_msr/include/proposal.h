#ifndef RPN_MSR_INCLUDE_PROPOSAL_H
#define RPN_MSR_INCLUDE_PROPOSAL_H

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
        class Proposal
        {
        public:
            Proposal(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     rpn_cls_prob_reshape, ::tensorflow::Input rpn_bbox_pred,
                     ::tensorflow::Input im_info, int64 cfg_key, int64 feat_stride);
            operator ::tensorflow::Output() const { return blob_tf; }
            operator ::tensorflow::Input() const { return blob_tf; }
            ::tensorflow::Node* node() const { return blob_tf.node(); }
            ::tensorflow::Output blob_tf;
        };
    }
}

#endif
