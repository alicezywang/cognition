#include "tensorrt_cognition_plugin/test_plugin.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(observe_softbus::TensorRTCognTest, general_bus::GeneralPlugin);
using namespace observe_softbus;
using namespace std;

/*******************************
 * classs TensorRTCognTest
 *******************************/
TensorRTCognTest::TensorRTCognTest()
{
    ROS_INFO("staring TensorRTCognTest Plugin ......");
}
TensorRTCognTest::~TensorRTCognTest(){
    ROS_INFO("TensorRTCognTest plugin had closed!");
}

void TensorRTCognTest::start()
{
    cognition::task_description task;  //include two structures: task_inf and actor_inf
    //Information from task_interface
    task.task_inf.task_id="001";
    task.task_inf.task_type = "Detection";    
    task.task_inf.battle_field = "air";  
    task.task_inf.target_category = "Car";   //target_info
    task.task_inf.target_status="static";   //target_info
    task.task_inf.target_pose = 60;  //the (x,y,z) axis of target, here is z-axis of target
    //Information from actor_interface
    task.actor_inf.actor_id="01";
    task.actor_inf.sensor_id="001";
    task.actor_inf.sensor_type = "visible";    
    task.actor_inf.cpu_info = "gpu";  
    task.actor_inf.memory_info = 0;  
    //根据任务的不同，认知总线 --> 初始化不同的认知资源
    cognition_handle = boost::make_shared<cognition::Handle>(cognition::UesType::Auto);
    cognition_handle->init(task);  // ptr must use "->" !!!!
    //ROS
    sub_img = pluginSubscribe("/recogImage", 1, &TensorRTCognTest::imgCallBack, this);
    pub_msg = pluginAdvertise<orient_softbus_msgs::RecogResult>("/machine_learning/detection", 1);
    pub_test = pluginAdvertise<sensor_msgs::Image>("/machine_learning/detection_test", 1);
}

void TensorRTCognTest::imgCallBack(const sensor_msgs::Image::ConstPtr& msg){
    //sensor_msgs::Image to cv::Mat
    cv::Mat cv_img;
    try
      {
        //cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        cv_img = cv_bridge::toCvCopy(msg, "bgr8")->image;//toCvCopy or toCvShare
      }
    catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }


    //调用认知资源
    vector<boost::any> inputs  = {cv_img,};
    vector<boost::any> results;
    //bool flag = cognition_handle->call(inputs, results);
    bool flag = cognition_handle->async_call(inputs, results, 3);

    if(flag){
      //output
      vector<string> vec_classes = boost::any_cast<vector<string>>(results[0]); //class names for one image
      vector<float>  vec_scores  = boost::any_cast<vector<float>>(results[1]);  //confidences of each class for one image
      vector<vector<float>> vec_bboxes = boost::any_cast<vector<vector<float>>>(results[2]); //bounding boxes of each class for one image
      //cv_img = boost::any_cast<cv::Mat>(results[3]);
      ROS_INFO("~~~~~~~~ Detection Class Num = %zu ~~~~~~~~", vec_classes.size());

      //封装msg
      orient_softbus_msgs::RecogResult ml_detection; //topic:"/machine_learning/detection"
      ml_detection.classes = vec_classes;   
      ml_detection.scores = vec_scores;   
      geometry_msgs::Polygon bbox; 
      int bb_num = vec_bboxes.size();         
      bbox.points.resize(2);
      for(int i=0; i< bb_num; ++i){
        bbox.points[0].x = vec_bboxes[i][0];
        bbox.points[0].y = vec_bboxes[i][1];
        bbox.points[1].x = vec_bboxes[i][2];
        bbox.points[1].y = vec_bboxes[i][3];
        ml_detection.bboxes.push_back(bbox);
      }
      std::cout << "~~~~~~~x = "<< bbox.points[0].x << std::endl;
      std::cout << "~~~~~~~y = "<< bbox.points[0].y << std::endl;
      std::cout << "~~~~~~~w = "<< bbox.points[1].x << std::endl;
      std::cout << "~~~~~~~h = "<< bbox.points[1].y << std::endl;
      
      //可视化
      for (int i = 0; i < vec_bboxes.size(); ++i) {
          int x = (int)vec_bboxes[i][0];
          int y = (int)vec_bboxes[i][1];
          int w = (int)vec_bboxes[i][2];
          int h = (int)vec_bboxes[i][3];
          cv::rectangle(cv_img, cv::Rect2d(x, y, w, h), cv::Scalar(255, 0, 0), 2, 1);
      }
  
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg(ml_detection.img_result);
      pub_msg.publish(ml_detection);
      pub_test.publish(ml_detection.img_result);
    }
    else {
        ROS_INFO("~~~~~~~~ GeneralPlugin/TensorRTCognTest : Detection Call Timeout ~~~~~~~~");
    }
}
