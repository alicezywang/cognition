#include "tensorflow/cc/ops/const_op.h"
#include "anchor_target.h"

namespace tensorflow
{
    namespace ops
    {
        AnchorTarget::AnchorTarget(const ::tensorflow::Scope& scope, ::tensorflow::Input rpn_cls_score,
                                   ::tensorflow::Input gt_boxes, ::tensorflow::Input im_info,
                                   ::tensorflow::Input data, int64 feat_stride)
        {
            if (!scope.ok()) return;
            auto _rpn_cls_score = ::tensorflow::ops::AsNodeOut(scope, rpn_cls_score);
            if (!scope.ok()) return;
            auto _gt_boxes = ::tensorflow::ops::AsNodeOut(scope, gt_boxes);
            if (!scope.ok()) return;
            auto _im_info = ::tensorflow::ops::AsNodeOut(scope, im_info);
            if (!scope.ok()) return;
            auto _data = ::tensorflow::ops::AsNodeOut(scope, data);
            if (!scope.ok()) return;
            ::tensorflow::Node* ret;
            const auto unique_name = scope.GetUniqueNameForOp("AnchorTarget");
            auto builder = ::tensorflow::NodeBuilder(unique_name, "AnchorTarget")
                           .Input(_rpn_cls_score)
                           .Input(_gt_boxes)
                           .Input(_im_info)
                           .Input(_data)
                           .Attr("feat_stride", feat_stride);
            scope.UpdateBuilder(&builder);
            scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
            if (!scope.ok()) return;
            scope.UpdateStatus(scope.DoShapeInference(ret));
            ::tensorflow::NameRangeMap _outputs_range;
            ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
            if (!_status_.ok())
            {
                scope.UpdateStatus(_status_);
                return;
            }
            this->rpn_labels = Output(ret, _outputs_range["rpn_labels"].first);
            this->rpn_bbox_targets = Output(ret, _outputs_range["rpn_bbox_targets"].first);
            this->rpn_bbox_inside_weights = Output(ret, _outputs_range["rpn_bbox_inside_weights"].first);
            this->rpn_bbox_outside_weights = Output(ret, _outputs_range["rpn_bbox_outside_weights"].first);
        }
    }
}
