#ifndef CognitionSoftBusDetection_H
#define CognitionSoftBusDetection_H

// C and C++ headers
#include <string>

// ROS headers
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// Interface headers
#include <cognition_bus/softbus_base.h>

/**
 * @brief namespace cognition_bus
 */
namespace cognition_bus {
using namespace std;

/**
 * @brief  The SoftBusDetection class of image target detection task
 *
 * @param  inputs: is a sequence of images
 *         inputs[0] <-- cv::Mat cv_img_;
 *
 * @return outputs: is result of image target detect
 *         outputs[0] <-- vector<string> vec_classes_;
 *         outputs[1] <-- vector<float>  vec_scores_;
 *         outputs[2] <-- vector<vector<float>> vec_bboxes_;
 */
class SoftBusDetection: public SoftBusBase{
public:
    virtual ~SoftBusDetection()=default;

    virtual bool call(vector<boost::any>& inputs, vector<boost::any>& results){}

protected:
    //input
    cv::Mat cv_img_;
    //output
    vector<string> vec_classes_;
    vector<float>  vec_scores_;
    vector<vector<float>> vec_bboxes_;
};

}//namespace


#endif // CognitionSoftBusDetection_H
