#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include <cstdlib>
#include "zero_out.h"
using namespace tensorflow;
using namespace tensorflow::ops;
tensorflow::Output zero_out(const tensorflow::Scope& scope, tensorflow::Input to_zero){
  return ZeroOut(scope, to_zero);
}
int main()
{
    sleep(3);
    std::vector<Tensor> outputs;
    Scope root = Scope::NewRootScope();
    ClientSession session(root);

    Variable x(root, {3,3}, DataType::DT_FLOAT);
    TF_CHECK_OK(session.Run({Assign(root, x, Multiply(root, TruncatedNormal(root, {3,3}, DataType::DT_FLOAT), Const(root, 1.0f)))}, NULL));

    //ZeroOut y(root, x);
    auto y = zero_out(root, x);
    TF_CHECK_OK(session.Run({y}, &outputs));
    std::cout << outputs[0].tensor<float, 2>() << std::endl;

    TF_CHECK_OK(session.Run({Multiply(root, y, Const(root, 0.1f))}, &outputs));
    std::cout << outputs[0].tensor<float, 2>() << std::endl;

    return 0;
}
