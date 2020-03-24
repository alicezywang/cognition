#include "cognition/cognition.h"
#include <boost/algorithm/string.hpp>

namespace cognition {

/**
 * @brief Handle::Handle
 * @param type
 * @note In Future, will add function calling of model_select and so on;
 */
Handle::Handle(UesType type )
    :type_(type)
{
}

Handle::~Handle(){
    if (UesType::Direct == type_ ) {
        ROS_INFO("Closed Cognition Handle Direct Mode");
    }
    else {
        impl_ptr_.reset();
        delete loader_;
        ROS_INFO("Closed Cognition Handle Auto Mode With Close <pluginlib>");
    }
}

//this function will get Task Param which is a c++ struct
void Handle::init(const cognition::task_description &task)
{
  if(UesType::Direct == type_) {
      ROS_INFO("Initialized Cognition Handle Direct Mode With getModelFromId()");
  }
  else if(UesType::Auto == type_)  {
    ROS_INFO("Auto Mode: Initialized Cognition Handle Auto Mode With getModelFromRecommend()");

    ModelMeta model = getModelFromRecommend(task);//HERE USE someparam
    string impl_name =  "cognition_bus::" + model.task_type + model.famework_type + "Impl"; 
    //"cognition_bus::DetectionTensorRTImpl"
    //ROS_INFO("Initialized Cognition Handle With Resource %s", impl_name.c_str());
    loader_ = new pluginlib::ClassLoader<cognition_bus::SoftBusBase>("cognition_bus","cognition_bus::SoftBusBase");
    impl_ptr_ = loader_->createInstance(impl_name);
    ROS_INFO("Initialized Cognition Handle With Resource %s", impl_name.c_str());
  }
  else {
    ROS_INFO("Initialized failed, Because wrong UesType");
  }
}

//UesType::Auto
bool Handle::call(vector<boost::any>& inputs, vector<boost::any>& results)
{
    return impl_ptr_->call(inputs, results);
}

bool Handle::async_call(vector<boost::any>& inputs, vector<boost::any>& results, int duration)
{
    return impl_ptr_->async_call(inputs, results, duration);
}

//TEST NEW FUNCTION
ModelMeta Handle::getModelFromRecommend(const cognition::task_description &task)
{
    //TODO
    //get all model of coginition resource
    std::map<std::string, cognition::model_description> model_list = cognition::load_model_list();
    //Return recommend model_list
    std::vector<std::string> result_list = cognition::get_recommended_models(task, model_list);
    //select the top_1 model to recomend
    string model_id = result_list[0]; //TODO : result_list MUST A model_id: TensorRT_MobileNetSSD_Car
    cout << "Recommend Result :" << result_list[0]<< endl;
    std::vector<std::string> resVec;
    boost::split(resVec, model_id, boost::is_any_of("_"), boost::token_compress_on);
    ModelMeta model;
    model.model_id = model_id;        // MUST <framework_name>_<model_name>_<dataset_name>
    model.famework_type = "TensorRT";  // MUST "TensorRT"
    model.task_type = task.task_inf.task_type; // MUST "Detection"
    return model;
}

//UesType::Direct
//get model path according model name(<framework_name>, <model_name>, <dataset_name>)
ModelMeta Handle::getModelFromName(const FrameworkType& framework_name, const string& model_name, const string &dataset_name)
{
    string pretrained_model_dir = ros::package::getPath("/pretrained_model"); 

    int idx = static_cast<int>(framework_name);
    string frame_name = FrameworkList_[idx];
    string model_root_path = pretrained_model_dir +"/"+ frame_name +"/"+ model_name +"/";

    //return
    ModelMeta modelmeta;
    modelmeta.model_root_path = model_root_path;
    return modelmeta;
}

//get model path according model_id(<framework_name>_<model_name>_<dataset_name>)
ModelMeta Handle::getModelFromId(const string& model_id)
{
    //splitWithPattern into resVec
    std::vector<std::string> resVec;
    boost::split(resVec, model_id, boost::is_any_of("_"), boost::token_compress_on);
    //std::string pattern = "_";
    //std::string strs = model_id + pattern; //方便截取最后一段数据
    //size_t pos = strs.find(pattern);
    //size_t size = strs.size();
    //while (pos != std::string::npos)
    //{
    //    std::string x = strs.substr(0,pos);
    //    resVec.push_back(x);
    //    strs = strs.substr(pos+1,size);
    //    pos = strs.find(pattern);
    //}

    //get getModelMeta
    string frame_name = resVec[0];
    string model_name = resVec[1];
    string dataset_name = resVec[2];

    string pretrained_model_dir = ros::package::getPath("/pretrained_model"); 
    string model_root_path = pretrained_model_dir +"/"+ frame_name +"/"+ model_name +"/";

    //return
    ModelMeta modelmeta;
    modelmeta.model_root_path = model_root_path;
    return modelmeta;
}

}//namespace
