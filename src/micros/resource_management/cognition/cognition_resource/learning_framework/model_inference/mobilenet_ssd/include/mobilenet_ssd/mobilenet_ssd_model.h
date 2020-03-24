#ifndef MOBILENET_SSD_MODEL_H
#define MOBILENET_SSD_MODEL_H

#include <opencv2/opencv.hpp>

#include "ml_model_base/model_base.h"
#include "mobilenet_ssd/pb_to_tf_network.h"

namespace machine_learning
{

class MobileNetSSDModel: ModelBase
{
public:
  MobileNetSSDModel();
  ~MobileNetSSDModel();
  virtual void train(int start, int end);
  virtual ResultType evaluate(cv::Mat &test_image);
  virtual void batch_evaluate(int size);
  std::vector<std::string> classNames = {"background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                                         "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                                         "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                                         "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
                                         "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"};
private:
  PbToTFNetwork network;
  Session *session;
};

} //machine_learning

#endif // MOBILENET_SSD_MODEL_H
