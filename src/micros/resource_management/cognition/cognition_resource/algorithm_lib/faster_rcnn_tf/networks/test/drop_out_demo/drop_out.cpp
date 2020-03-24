#include "tensorflow/cc/ops/const_op.h"
#include "drop_out.h"

namespace tensorflow {
namespace ops {

DropOut::DropOut(const ::tensorflow::Scope& scope, ::tensorflow::Input to_drop) {
  if (!scope.ok()) return;
  auto _to_drop = ::tensorflow::ops::AsNodeOut(scope, to_drop);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("DropOut");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "DropOut")
                     .Input(_to_drop)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

}
}
