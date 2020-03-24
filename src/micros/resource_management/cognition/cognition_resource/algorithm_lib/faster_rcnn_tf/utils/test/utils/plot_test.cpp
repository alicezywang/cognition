#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/OpenCVPlot.h"

int main()
{
    std::string text = "test text";
    cv::Mat imag = cv::imread("/home/yan/image_test/src/drawrect/004545.jpg", CV_LOAD_IMAGE_COLOR);
    std::vector<float> pos1;
    float a =1.0;
    pos1.push_back(a);
    pos1.push_back(a*80);
    pos1.push_back(a*160);
    pos1.push_back(a*240);

    drawRectOnImage(imag, pos1);
    drawTextOnImage(imag, pos1, text);
    cv::imshow("Result", imag);
    cvWaitKey(0);
    return 0;
}
