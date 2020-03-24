#ifndef PB_TO_DNN_NETWORK_H
#define PB_TO_DNN_NETWORK_H

#include <opencv2/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace machine_learning {

using namespace std;
using namespace cv;

class PbToDNNNetwork
{
public:
  PbToDNNNetwork();
  ~PbToDNNNetwork();
  dnn::Net *net;
};

} // machine_learning

#endif // PB_TO_DNN_NETWORK_H
