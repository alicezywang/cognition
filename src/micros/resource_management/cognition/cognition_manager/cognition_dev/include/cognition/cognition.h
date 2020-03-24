#ifndef CognitionHandle_H
#define CognitionHandle_H

// C and C++ headers
#include <string>
#include <vector>
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

// ROS headers
#include <ros/ros.h>
#include <pluginlib/class_loader.h>

// Interface headers
#include <cognition_bus/softbus_base.h>

#include <model_recommend/recommend.h>
/**
 * @brief namespace cognition
 */
namespace cognition {
using namespace std;

enum class UesType{
    Auto,
    Direct,
};

enum class FrameworkType{
    Caffe,
    DNN,
    TensorRT,
    TensorFlow,
//    NCNN,
//    Pytorch,
};

struct ModelMeta{
    string model_id;       // = "TensorRT_MobileNetSSD_Car"
    string maintainer;     // = "observe"
    string task_type;      // = "Detection"
    string famework_type;  // = "Caffe"
    string model_root_path;// model_root_path
};

/**
 * @brief The interface class of cognition Handle
 */
class Handle{
public:
    Handle(UesType type = UesType::Auto);
    ~Handle();

    //init param
    void init(const cognition::task_description &task);
    void init(const string& model_id);

    //UesType::Direct
    ModelMeta getModelFromName(const FrameworkType& framework_name, const string& model_name, const string& dataset_name);
    ModelMeta getModelFromId(const string& model_id);

    //UesType::Auto
    ModelMeta getModelFromRecommend(const cognition::task_description &task);
    bool call(vector<boost::any>& inputs, vector<boost::any>& results);
    bool async_call(vector<boost::any>& inputs, vector<boost::any>& results, int duration = 10);
    //ModelMeta model_evaluate(const cognition::task_description &task,std::vector<std::string> result_list);

private:
    UesType type_;

    pluginlib::ClassLoader<cognition_bus::SoftBusBase>* loader_; // Pointer to Bus loader
    boost::shared_ptr<cognition_bus::SoftBusBase> impl_ptr_;    // Pointer to loaded class

    vector<string> FrameworkList_ = {
        "Caffe",
        "DNN",
        "TensorRT",
        "TensorFlow",
    };
};


}//namespace


#endif // CognitionHandle_H
