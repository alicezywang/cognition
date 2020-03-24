#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "time_op.h"

using namespace tensorflow;

int main(int argc, char **argv)
{
  std::vector<Tensor> outputs;
  Scope scope = Scope::NewRootScope();
  ClientSession session(scope);
  auto time1 = tensorflow::ops::Time(scope.WithOpName("time1"), 0.0f);
  auto time2 = tensorflow::ops::Time(scope.WithOpName("time2"), 0.0f);
  session.Run({time1.output_data, time1.time}, &outputs);
  std::cout<<std::fixed<<outputs[0].scalar<float>()<<std::endl;
  std::cout<<std::fixed<<outputs[1].scalar<double>()<<std::endl;
  std::cout<<"----------------end--1!-------------"<<std::endl;
  session.Run({time2.output_data, time2.time}, &outputs);
  std::cout<<std::fixed<<outputs[0].scalar<float>()<<std::endl;
  std::cout<<std::fixed<<outputs[1].scalar<double>()<<std::endl;
  std::cout<<"----------------end--2!-------------"<<std::endl;
  return 0;
}
