#include "tensorflow/cc/ops/const_op.h"
#include "proposal.h"

namespace tensorflow
{
    namespace ops
    {
        Proposal::Proposal(const ::tensorflow::Scope& scope, ::tensorflow::Input
                           rpn_cls_prob_reshape, ::tensorflow::Input rpn_bbox_pred,
                           ::tensorflow::Input im_info, int64 cfg_key, int64 feat_stride)
        {
            if (!scope.ok()) return;
            auto _rpn_cls_prob_reshape = ::tensorflow::ops::AsNodeOut(scope, rpn_cls_prob_reshape);
            if (!scope.ok()) return;
            auto _rpn_bbox_pred = ::tensorflow::ops::AsNodeOut(scope, rpn_bbox_pred);
            if (!scope.ok()) return;
            auto _im_info = ::tensorflow::ops::AsNodeOut(scope, im_info);
            if (!scope.ok()) return;
            ::tensorflow::Node* ret;
            const auto unique_name = scope.GetUniqueNameForOp("Proposal");
            auto builder = ::tensorflow::NodeBuilder(unique_name, "Proposal")
                            .Input(_rpn_cls_prob_reshape)
                            .Input(_rpn_bbox_pred)
                            .Input(_im_info)
                            .Attr("cfg_key", cfg_key)
                            .Attr("feat_stride", feat_stride);
            scope.UpdateBuilder(&builder);
            scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
            if (!scope.ok()) return;
            scope.UpdateStatus(scope.DoShapeInference(ret));
            this->blob_tf = Output(ret, 0);
        }
    }
}
