#ifndef NETWORK_H
#define NETWORK_H

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include <string>
#include <vector>
#include <map>

using namespace tensorflow::ops;

using tensorflow::Scope;
using tensorflow::Input;
using tensorflow::Output;
using tensorflow::Tensor;
using tensorflow::DT_INT32;
using tensorflow::DT_FLOAT;
using tensorflow::GraphDef;
using tensorflow::OutputHash;
using tensorflow::TensorShape;
using tensorflow::MetaGraphDef;
using tensorflow::ClientSession;

typedef std::unordered_map<Output, Input::Initializer, OutputHash> FeedType;

class Network
{
protected:
  std::vector<Output> inputs;
  std::vector<Output> trainable_variables;
  std::vector<Output> non_trainable_variables;
  std::vector<TensorShape> trainable_variable_shapes;
  std::map<std::string, std::vector<Output> > layers;
  std::map<std::string, Output> weights;
  std::map<std::string, Output> biases;
  std::map<std::string, TensorShape> weight_shapes;
  std::map<std::string, TensorShape> bias_shapes;

public:
  std::vector<Output> timers;
  Scope scope;
  ClientSession session;
  FeedType feed_map;
  Network(const Scope &scope);
  ~Network();
  virtual void setup() = 0;
  Network *feed(const std::vector<std::string> &names);
  Network *load(ClientSession &session, std::string file_path);
  Network *add_output(std::string name, Output op);
  const std::vector<Output> &get_output(std::string name);
  const std::vector<Output> &get_trainable_variables();
  const std::vector<Output> &get_non_trainable_variables();
  const std::vector<TensorShape> &get_trainable_variable_shapes();
  bool output_exists(std::string name);
  std::vector<std::string> get_weight_list();
  std::vector<std::string> get_bias_list();
  GraphDef get_graph_def();
  Output get_weight(std::string name, TensorShape shape = TensorShape(), bool trainable = true);
  Output get_bias(std::string name, TensorShape shape = TensorShape(), bool trainable = true);
  bool validate_padding(std::string padding);
  Network *conv(int k_h, int k_w, int c_o, int s_h, int s_w, std::string name, bool relu = true, std::string padding = "SAME", int group = 1, bool trainable = true);
  Network *relu(std::string name);
  Network *max_pool(int k_h, int k_w, int s_h, int s_w, std::string name, std::string padding);
  Network *avg_pool(int k_h, int k_w, int s_h, int s_w, std::string name, std::string padding);
  Network *roi_pool(int pooled_height, int pooled_width, float spatial_scale, std::string name);
  Network *roi_align(int pooled_height, int pooled_width, float spatial_scale, std::string name);
  Network *proposal_layer(const std::vector<int> &_feat_stride, const std::vector<int> &anchor_scales, std::string cfg_key, std::string name);
  Network *anchor_target_layer(const std::vector<int> &_feat_stride, const std::vector<int> &anchor_scales, std::string name);
  Network *proposal_target_layer(int classes, std::string name);
  Network *reshape_layer(int d, std::string name);
  Network *lrn(int radius, float alpha, float beta, std::string name, float bias);
  Network *fc(int num_out, std::string name, bool relu = true, bool trainable = true);
  Network *softmax(std::string name);
  Network *dropout(float keep_prob, std::string name);
  Network *timer(std::string name);
  Network *timer_leaf(std::string name);
};

#endif
