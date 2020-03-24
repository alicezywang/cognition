#include <ros/package.h>
#include "mobilenet_ssd/pb_to_tf_network.h"

namespace machine_learning {

PbToTFNetwork::PbToTFNetwork()
{
  std::string pbPath = ros::package::getPath("pretrained_model") + "/TensorFlow/ssd/frozen_inference_graph.pb";
  ReadBinaryProto(Env::Default(), pbPath, &graph_def);
}

PbToTFNetwork::~PbToTFNetwork()
{

}

} //machine_learning
