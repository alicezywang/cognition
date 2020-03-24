#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

#include "roi_pool.h"

using namespace tensorflow;

Status RoiPoolGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  int64 pooled_height;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "pooled_height", &pooled_height));
  int64 pooled_width;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "pooled_width", &pooled_width));
  float spatial_scale;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "spatial_scale", &spatial_scale));
  grad_outputs->push_back(tensorflow::ops::RoiPoolGrad(scope, op.input(0), op.input(1), op.output(1), grad_inputs[0], pooled_height, pooled_width, spatial_scale));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

Status CastGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

Status SparseSoftmaxCrossEntropyWithLogitsGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(tensorflow::ops::Multiply(scope, tensorflow::ops::ExpandDims(scope, grad_inputs[0], -1), op.output(1)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("RoiPool", RoiPoolGrad);
REGISTER_GRADIENT_OP("Cast", CastGrad);
REGISTER_GRADIENT_OP("SparseSoftmaxCrossEntropyWithLogits", SparseSoftmaxCrossEntropyWithLogitsGrad);
