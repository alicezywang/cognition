#ifndef _OPENCVPLOT_H
#define _OPENCVPLOT_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void drawRectOnImage(cv::Mat& im, std::vector<float> pos, float thickness = 2.0/*线框粗细*/);
float drawTextOnImage(cv::Mat& im, std::vector<float> pos, std::string text, float voffset = 0.0 /*文本在框中的位置调节参数*/,  float thickness = 1.0 /*文本粗细*/, float fontScale = 0.5 /*文本大小*/);

#endif