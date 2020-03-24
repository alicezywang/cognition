#ifndef CognitionDetectionDNNImpl_H
#define CognitionDetectionDNNImpl_H

// C and C++ headers
#include <string>

// ROS headers
#include <ros/ros.h>
#include <ros/package.h>

// Interface headers
#include <cognition_bus/softbus_detection.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <stdint.h>


/**
 * @brief namespace cognition_bus
 */
namespace cognition_bus {
using namespace std;


class DetectionDNNImpl: public SoftBusDetection{
public:
  DetectionDNNImpl();
 /**
 * @brief  The DetectionDNNImpl class of image target detection task
 *
 * @param  inputs: is a sequence of images
 *         inputs[0] <-- cv::Mat cv_img_;
 *
 * @return outputs: is result of image target detect
 *         outputs[0] <-- vector<string> vec_classes_;
 *         outputs[1] <-- vector<float>  vec_scores_;
 *         outputs[2] <-- vector<vector<float>> vec_bboxes_;
 */
  virtual bool call(vector<boost::any>& inputs, vector<boost::any>& results);

private:
  cv::dnn::Net net;
  float confidenceThreshold;

  const size_t inWidth = 224;
  const size_t inHeight = 224;
  const float inScaleFactor = 0.007843f;
  const float meanVal = 127.5;
  /*const char* classNames[21] = {"background", "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                                "train", "tvmonitor"};*/
  const char* classNames[21] = {"background", "car", "bus", "trunk", "motor",
                                "bike", "tanker"};
 	const cv::String keys = ("{ help           | false | print usage         }"
         "{ proto          | MobileNetSSD_deploy.prototxt   | model configuration }"
         "{ model          | MobileNetSSD_deploy.caffemodel | model weights }"
         "{ camera_device  | 0     | camera device number }"
         "{ camera_width   | 640   | camera device width  }"
         "{ camera_height  | 480   | camera device height }"
         "{ video          |       | video or image for detection}"
         "{ out            |       | path to output video file}"
         "{ min_confidence | 0.2   | min confidence      }"
         "{ opencl         | false | enable OpenCL }");

}; // class DetectionDNNImpl

}//namespace


#endif // CognitionDetectionDNNImpl_H
