#include <utility>
#include "detection_dnn_impl/detection_dnn_impl.h"

namespace cognition_bus {

DetectionDNNImpl::DetectionDNNImpl()
{
  string pretrained_model_dir = ros::package::getPath("/pretrained_model"); 
  string netcfg_path_ = pretrained_model_dir + "/Caffe/MobileNetSSD/Caffe_MobileNetSSD_CoCo.prototxt";
  string model_path_ = pretrained_model_dir + "/Caffe/MobileNetSSD/Caffe_MobileNetSSD_CoCo.caffemodel";
  cv::String modelConfiguration = netcfg_path_;
  cv::String modelBinary = model_path_;
  
  net = cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary);

  // Confidence threshold of the detected objects
  confidenceThreshold = 0.1;
}


/**
 * @brief  DetectionTFImpl::call
 *
 * @param  inputs: is a sequence of images
 *         inputs[0] <-- cv::Mat cv_img_;
 *
 * @return outputs: is result of image target detect
 *         outputs[0] <-- vector<string> vec_classes_;
 *         outputs[1] <-- vector<float>  vec_scores_;
 *         outputs[2] <-- vector<vector<float>> vec_bboxes_;
 */
bool DetectionDNNImpl::call(vector<boost::any>& inputs, vector<boost::any>& results)
{
    //inputs
    cv_img_ = boost::any_cast<cv::Mat>(inputs[0]);
    //outputs
    //vec_classes_;
    //vec_scores_;
    //vec_bboxes_;

  // Prepare blob: convert cv::Mat to batch of images
  cv::Mat inputBlob = cv::dnn::blobFromImage(cv_img_, inScaleFactor, cv::Size(inWidth, inHeight), 
                                           cv::Scalar(meanVal, meanVal, meanVal), false, false);
  // Set input blob: set the network input
  net.setInput(inputBlob);
  // Make forward pass: inference bounding boxes
  cv::Mat detection = net.forward();
 
  // Scan through all deteced objects
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

  
  for(int i = 0; i < detectionMat.rows; i++)
  {
    float confidence = detectionMat.at<float>(i, 2);
    if(confidence > confidenceThreshold)
    {
      size_t objectClassID = (size_t)(detectionMat.at<float>(i, 1));
      // Calculate pixel coordinates
      float left = detectionMat.at<float>(i, 3) * cv_img_.cols;
      float top = detectionMat.at<float>(i, 4) * cv_img_.rows;
      float right = detectionMat.at<float>(i, 5) * cv_img_.cols;
      float bottom = detectionMat.at<float>(i, 6) * cv_img_.rows;
      // Filter non-car objects
      if (classNames[objectClassID] == "car")
      { 
        vec_classes_.push_back("car");
        vec_scores_.push_back(confidence);
        //cv::Rect rt(left, top, right - left, bottom - top);
        vector<float> bbox;
        bbox.push_back(left);  
        bbox.push_back(top);
        bbox.push_back(right - left);  
        bbox.push_back(bottom - top);
        vec_bboxes_.push_back(bbox);
      }
    }
  }

  //output
  results.push_back(std::move(vec_classes_));
  results.push_back(std::move(vec_scores_));
  results.push_back(std::move(vec_bboxes_));

  return true;
}

}//namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(cognition_bus::DetectionDNNImpl, cognition_bus::SoftBusBase)
