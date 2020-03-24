#include <ros/package.h>
#include "mobilenet_ssd/pb_to_dnn_network.h"

namespace machine_learning {

PbToDNNNetwork::PbToDNNNetwork()
{
  String weights = ros::package::getPath("pretrained_model")  + "/TensorFlow/ssd/frozen_inference_graph.pb";
  String prototxt = ros::package::getPath("pretrained_model") + "/TensorFlow/ssd/ssd_mobilenet_v1_coco.pbtxt";
  net = new dnn::Net(cv::dnn::readNetFromTensorflow(weights, prototxt));
}

PbToDNNNetwork::~PbToDNNNetwork()
{
  delete net;
}

} //machine_learning
