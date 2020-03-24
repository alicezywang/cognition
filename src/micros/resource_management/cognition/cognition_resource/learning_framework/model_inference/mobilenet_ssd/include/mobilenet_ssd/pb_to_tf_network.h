#ifndef PB_TO_TF_NETWORK_H
#define PB_TO_TF_NETWORK_H

#include <string>
#include <vector>
#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/image_ops.h>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

//opencv
#include <opencv2/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace machine_learning {

using namespace std;
using namespace cv;
using namespace tensorflow;
using namespace tensorflow::ops;

class PbToTFNetwork
{
public:
  PbToTFNetwork();
  ~PbToTFNetwork();
  GraphDef graph_def;
};

} //namespace machine_learning

#endif // PB_TO_TF_NETWORK_H
