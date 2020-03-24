#ifndef IMGDATA_H
#define IMGDATA_H
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <map>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tensorflow/core/framework/tensor.h>
#include "plastic_net/config.h"

using namespace cv;
/**
 * @brief The ImgData class
 */
class ImgData
{
public:
  ImgData(std::string ParamDir);
  ~ImgData();
  void GetImgPath(std::string Input_param_dir);
  std::multimap<std::string, std::vector<std::string> > GetClassinfo();
  ///SplitString "/"
  std::vector<std::string> SplitString(const std::string &s, const std::string &c);
  ///reade image
  std::pair<Mat, std::vector<std::string> > RandomSomeoneImgData();     //Random ImageData interface API
  cv::Mat rotate_arbitrarily_angle(Mat &src, int Inputangle);
  std::pair<Mat, std::vector<std::string> > rotate_image();             // rotate ImageData Interface API

  cv::Mat rotate(int rotate, Mat src);
  tensorflow::Tensor OutputImageData(std::pair<int, std::string> imageinfo);

  int filesnumber;
  std::string Input_param;
  std::vector<std::string> ImgPathList;                             //path
  std::vector<std::string> temp_imagePath;
  std::vector<std::vector<std::string> > imagedata;                      //{20}
  std::vector<std::vector<std::string> > fixedDate;
  std::vector<std::vector<std::string> > ImgClassinfo;                   //SplitString later
  std::multimap<std::string, std::vector<std::string> > ImgPathToClassInfo;    //interface API
};

/**
 * @brief generateInputsLabelsAndTarget
 * @param params
 * @param imagedata
 * @param is_test
 * @return
 */
std::vector<tensorflow::Tensor> generateInputsLabelsAndTarget(const DefaultParams &params, ImgData &imagedata);

#endif // IMGDATA_H
