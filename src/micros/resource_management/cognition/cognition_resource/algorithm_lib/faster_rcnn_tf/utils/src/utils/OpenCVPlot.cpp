#include "utils/OpenCVPlot.h"

void drawRectOnImage(cv::Mat& im, std::vector<float> pos, float thickness) 
{    
    cv::Scalar color(255, 0, 0);// 颜色为蓝色.

    cv::Mat overlay;
    im.copyTo(overlay);
    //std::cout <<"pos[0:3] = "
    //     << pos[0] << ","
    //     << pos[1] << ","
    //     << pos[2] << ","
    //     << pos[3] << "," << std::endl;
    // cv::rectangle(overlay, cvPoint(pos[0], pos[1]),
    //                        cvPoint(pos[2] + pos[0], pos[3] + pos[1]),
    //               color, thickness);//画框

    cv::rectangle(overlay, cvPoint(pos[0], pos[1]), cvPoint(pos[2], pos[3]), color, thickness);
    float alpha = 1;//融合比例
    cv::addWeighted(overlay, alpha, im, 1 - alpha, 0, im);//实现图像线性混合
}

float drawTextOnImage(cv::Mat& im, std::vector<float> pos, std::string text, float voffset, float thickness, float fontScale) 
{
    int fontFace = cv::FONT_HERSHEY_DUPLEX;//字体
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);//获取待绘制文本框的大小，返回一个指定字体以及大小的string所占的空间

    cv::Mat overlay;
    im.copyTo(overlay);

    cv::rectangle(overlay, cvPoint(pos[0], pos[1] - thickness + voffset),
                           cvPoint(pos[0] + textSize.width, pos[1] - textSize.height - thickness + voffset),
                  cv::Scalar(0, 0, 255), CV_FILLED);//画框

    cv::Point textOrg(pos[0] , pos[1] - thickness + voffset);//文本框的左下角
    cv::putText(overlay, text, textOrg,
        fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);//在图像上绘图
    float alpha = 0.62;//融合比例
    cv::addWeighted(overlay, alpha, im, 1 - alpha, 0, im);//实现图像线性混合
    return float(textSize.height) + thickness;
}
