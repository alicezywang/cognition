#include "tensorflow/cc/ops/const_op.h"
#include "time_op.h"

namespace tensorflow {
namespace ops {

Time::Time(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_data) {
  if (!scope.ok()) return;
  auto _input_data = ::tensorflow::ops::AsNodeOut(scope, input_data);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Time");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "Time")
                     .Input(_input_data)
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

  this->output_data = Output(ret, _outputs_range["output_data"].first);
  this->time = Output(ret, _outputs_range["time"].first);
}

}
}
