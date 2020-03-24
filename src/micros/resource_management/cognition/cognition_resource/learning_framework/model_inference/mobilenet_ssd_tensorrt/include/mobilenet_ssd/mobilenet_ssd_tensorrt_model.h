#ifndef MOBILENET_SSD_TENSORRT_MODEL_H
#define MOBILENET_SSD_TENSORRT_MODEL_H

// C and C++ headers
#include <string>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ROS headers
#include <ros/ros.h>
#include <ros/package.h>

//CUDA headers
#include <cuda.h>

// Interface headers
#include <ml_model_base/model_base.h>

#include "mobilenet_ssd/tensorrt/common.h"
#include "mobilenet_ssd/tensorrt/tensorNet.h"
#include "mobilenet_ssd/tensorrt/imageBuffer.h"
#include "mobilenet_ssd/tensorrt/util/cuda/cudaMappedMemory.h"

namespace machine_learning
{

class MobileNetSSDTensorRTModel : public ModelBase
{
public:
  MobileNetSSDTensorRTModel();
  ~MobileNetSSDTensorRTModel();
  virtual void train(int start, int end);
  virtual ResultType evaluate(cv::Mat &cv_img);
  virtual void batch_evaluate(int size);

private:
  float* _allocateMemory(DimsCHW dims, char* info);
  void _loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale );

  //TensorRT
  TensorNet tensorNet;

  const char* INPUT_BLOB_NAME;
  const char* OUTPUT_BLOB_NAME;
  std::vector<std::string> output_vector;

  //void* imgCPU;
  //void* imgCUDA;
  void* imgCPU = nullptr;
  void* imgCUDA = nullptr;
  //float* data;
  float* output;

  const size_t inWidth = 300;
  const size_t inHeight = 300;

  const float meanVal = 127.5;
  /*const char* classNames[21] = {"background", "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                                "train", "tvmonitor"};*/
  const char* classNames[7] = {"background", "car", "car", "trunk", "motor",
                                "bike", "car"};
  const cv::String keys = ("{ help           | false | print usage         }"
         "{ proto          | MobileNetSSD_deploy.prototxt   | model configuration }"
         "{ model          | MobileNetSSD_deploy.caffemodel | model weights }"
         "{ camera_device  | 0     | camera device number }"
         "{ camera_width   | 640   | camera device width  }"
         "{ camera_height  | 480   | camera device height }"
         "{ video          |       | video or image for detection}"
         "{ out            |       | path to output video file}"
         "{ min_confidenCognitionDetectionTensorrtImpl_Hce | 0.2   | min confidence      }"
         "{ opencl         | false | enable OpenCL }");

};


class Timer {
public:
    void tic()
    {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc()
    {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //ROS_INFO("Time: %f ms", t);
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

} //machine_learning

#endif // MOBILENET_SSD_DNN_MODEL_H
