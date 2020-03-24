#include "tensorflow/cc/ops/const_op.h"
#include "roi_pool.h"

namespace tensorflow {
namespace ops {

RoiPool::RoiPool(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 bottom_data, ::tensorflow::Input bottom_rois, int64
                 pooled_height, int64 pooled_width, float spatial_scale) {
  if (!scope.ok()) return;
  auto _bottom_data = ::tensorflow::ops::AsNodeOut(scope, bottom_data);
  if (!scope.ok()) return;
  auto _bottom_rois = ::tensorflow::ops::AsNodeOut(scope, bottom_rois);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("RoiPool");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "RoiPool")
                     .Input(_bottom_data)
                     .Input(_bottom_rois)
                     .Attr("pooled_height", pooled_height)
                     .Attr("pooled_width", pooled_width)
                     .Attr("spatial_scale", spatial_scale)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  ::tensorflow::NameRangeMap _outputs_range;
  ::tensorflow::Status _status_ = ::tensorflow::NameRangesForNode(*ret, ret->op_def(), nullptr, &_outputs_range);
  if (!_status_.ok()) {
    scope.UpdateStatus(_status_);
    return;
  }

  this->top_data = Output(ret, _outputs_range["top_data"].first);
  this->argmax = Output(ret, _outputs_range["argmax"].first);
}

RoiPoolGrad::RoiPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         bottom_data, ::tensorflow::Input bottom_rois,
                         ::tensorflow::Input argmax, ::tensorflow::Input grad,
                         int64 pooled_height, int64 pooled_width, float
                         spatial_scale) {
  if (!scope.ok()) return;
  auto _bottom_data = ::tensorflow::ops::AsNodeOut(scope, bottom_data);
  if (!scope.ok()) return;
  auto _bottom_rois = ::tensorflow::ops::AsNodeOut(scope, bottom_rois);
  if (!scope.ok()) return;
  auto _argmax = ::tensorflow::ops::AsNodeOut(scope, argmax);
  if (!scope.ok()) return;
  auto _grad = ::tensorflow::ops::AsNodeOut(scope, grad);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("RoiPoolGrad");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "RoiPoolGrad")
                     .Input(_bottom_data)
                     .Input(_bottom_rois)
                     .Input(_argmax)
                     .Input(_grad)
                     .Attr("pooled_height", pooled_height)
                     .Attr("pooled_width", pooled_width)
                     .Attr("spatial_scale", spatial_scale)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

}
}
