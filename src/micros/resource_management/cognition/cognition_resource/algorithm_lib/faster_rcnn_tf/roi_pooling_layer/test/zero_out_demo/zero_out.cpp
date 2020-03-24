#include "tensorflow/cc/ops/const_op.h"
#include "zero_out.h"

namespace tensorflow {
namespace ops {

ZeroOut::ZeroOut(const ::tensorflow::Scope& scope, ::tensorflow::Input to_zero) {
  if (!scope.ok()) return;
  auto _to_zero = ::tensorflow::ops::AsNodeOut(scope, to_zero);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("ZeroOut");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "ZeroOut")
                     .Input(_to_zero)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->output = Output(ret, 0);
}

}
}
