#include "cognition_bus/detection_tensorrt_impl.h"

namespace cognition_bus {

DetectionTensorRTImpl::DetectionTensorRTImpl()
{
    string name = "MobileNetSSD";

    if(name == "MobileNetSSD"){
        ml_model = boost::make_shared<machine_learning::MobileNetSSDTensorRTModel>();
        cout <<"~~~ MobileNetSSD Task Starting ~~~"<< endl;
    }
    else if (name == "MASK") {
        //ml_model = boost::make_shared<machine_learning::MaskRCNNModel>();
        cout <<"~~~ MASK RCNN Task Starting ~~~"<< endl;
    }

}
/**
 * @brief  DetectionTensorRTImpl::call
 *
 * @param  inputs: is a sequence of images
 *         inputs[0] <-- cv::Mat cv_img_;
 *
 * @return outputs: is result of image target detect
 *         outputs[0] <-- vector<string> vec_classes_;
 *         outputs[1] <-- vector<float>  vec_scores_;
 *         outputs[2] <-- vector<vector<float>> vec_bboxes_;
 */
bool DetectionTensorRTImpl::call(vector<boost::any>& inputs, vector<boost::any>& results)
{

    //inputs
    cv::Mat _cv_img_ = boost::any_cast<cv::Mat>(inputs[0]);
    //sleep(2);
    //outputs
    //vec_classes_;
    //vec_scores_;
    //vec_bboxes_;

    //core
    machine_learning::ResultType result = ml_model->evaluate(_cv_img_);

    //return output
    results.push_back(std::move(result.vec_classes));  
    results.push_back(std::move(result.vec_scores));   
    results.push_back(std::move(result.vec_bboxes));   
    results.push_back(std::move(_cv_img_));
    return true;
}

}//namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(cognition_bus::DetectionTensorRTImpl, cognition_bus::SoftBusBase)
