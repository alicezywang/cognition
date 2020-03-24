#ifndef PLASTIC_NET_MODEL_H
#define PLASTIC_NET_MODEL_H

#include <iostream>
#include <iomanip>

#include <tensorflow/core/kernels/training_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/gradients.h>

//#include <tensorflow/cc/client/client_session.h>
//#include <tensorflow/cc/ops/standard_ops.h>
//#include <tensorflow/core/protobuf/meta_graph.pb.h>
//#include <tensorflow/core/platform/default/logging.h>

#include "ml_model_base/model_base.h"
#include "plastic_net/random.h"
#include "plastic_net/imgdata.h"
#include "plastic_net/config.h"
#include "plastic_net/plastic_network.h"

namespace machine_learning
{

using namespace tensorflow::ops;

class PlasticNetModel: ModelBase
{
private:
  const DefaultParams params;
  tensorflow::Scope scope;
  tensorflow::ClientSession session;
  std::string ml_datasets;
  std::string pretrained_model;
  Network net;

  void snapshot(int iter);
  void restore(std::string filename);

public:
  PlasticNetModel(const tensorflow::Scope &scope);
  ~PlasticNetModel();

  virtual void train(int start, int end);
  virtual ResultType evaluate(cv::Mat &test_image);
  virtual void batch_evaluate(int size);
};

} //machine_learning

#endif // PLASTIC_NET_MODEL_H
