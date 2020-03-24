#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "generate_anchors.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/core/graph/default_device.h"


using namespace tensorflow;

int main(int argc, char **argv)
{
//  SessionOptions options;
//  std::unique_ptr<Session> session(NewSession(options));

  Scope root = Scope::NewRootScope();
  ClientSession session(root);

  // create tensorflow structure start
  auto rpn_cls_score = ops::Variable(root,{},DT_FLOAT);
  auto assign_W = tensorflow::ops::Assign(root.WithOpName("rpn_cls_score"), rpn_cls_score, tensorflow::ops::RandomNormal(root, {5,5,5,5}, DT_FLOAT));

  auto gt_boxes = ops::Variable(root, {},DT_FLOAT);
  auto assign_b = tensorflow::ops::Assign(root.WithOpName("gt_boxes"), gt_boxes, tensorflow::ops::RandomNormal(root, {1,1}, DT_FLOAT));

  auto data = ops::Variable(root, {},DT_FLOAT);
  auto assign_c = tensorflow::ops::Assign(root.WithOpName("data"), data, tensorflow::ops::RandomNormal(root, {1,1,1,1}, DT_FLOAT));

  auto im_info = ops::Variable(root, {},DT_FLOAT);
  auto assign_d = tensorflow::ops::Assign(root.WithOpName("im_info"), im_info, tensorflow::ops::RandomNormal(root, {5,5}, DT_FLOAT));

  tensorflow::int64 feat_stride=1;

  auto anchor_target = tensorflow::ops::AnchorTarget(root.WithOpName("result"),rpn_cls_score,gt_boxes,im_info,data,feat_stride);


  //std::cout<<anchor_target.rpn_bbox_inside_weights.name()<<std::endl;

//  GraphDef graph_def;
//  root.ToGraphDef(&graph_def);
//  session->Create(graph_def);

  std::cout<<"Create ok"<<std::endl;

  //tensor input
  Tensor a(DT_FLOAT,TensorShape({1,1,1,1}));

  Tensor b(DT_FLOAT,TensorShape({1,1}));

  Tensor c(DT_FLOAT,TensorShape({1,1,1,1}));

  Tensor d(DT_FLOAT,TensorShape({1,1}));


  std::unordered_map<tensorflow::Output, Input::Initializer, tensorflow::OutputHash> inputs = {
    {rpn_cls_score,a},
    {gt_boxes,b},
    {data,c},
    {im_info,d},
  };

  std::vector<Tensor>outputs;
  TF_CHECK_OK(session.Run(inputs, {anchor_target.rpn_bbox_inside_weights,anchor_target.rpn_bbox_outside_weights,anchor_target.rpn_bbox_targets,anchor_target.rpn_labels}, &outputs));

  std::cout<<"Run ok"<<std::endl;
  //auto out_y = outputs[0].scalar<float>();
  std::cout <<outputs.size()<<std::endl;
  std::cout <<anchor_target.rpn_bbox_inside_weights.name()<<std::endl;
  //std::cout <<out_y<<std::endl;
//  session->Close();

  std::cout<<"----------------end!-------------"<<std::endl;
}
