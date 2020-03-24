#ifndef CognitionDetectionTFImpl_H
#define CognitionDetectionTFImpl_H

// C and C++ headers
#include <string>

// ROS headers
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// Interface headers
#include <cognition_bus/softbus_detection.h>
#include <fast_rcnn/faster_rcnn_model.h>

/**
 * @brief namespace cognition_bus
 */
namespace cognition_bus {
using namespace std;

/**
 * @brief The DetectionTFImpl class of image target detection task
 */
class DetectionTFImpl final : public SoftBusDetection{
public:
    DetectionTFImpl();
    /**
     * @brief  The call function is creaded for image target detection task
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

protected:
    boost::shared_ptr<machine_learning::ModelBase> ml_model; //for different ml task

};

}//namespace


#endif // CognitionDetectionTFImpl_H
