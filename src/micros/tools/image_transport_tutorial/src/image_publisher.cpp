#include <ros/ros.h>
#include <image_transport/image_transport.h>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~"); // 私有空间命名
  std::string pub_topic = "/image_raw";
  float pub_rate = 5;
  private_nh.param("pubTopic", pub_topic, pub_topic);
  private_nh.param("pubRate",  pub_rate,  pub_rate);

  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise(pub_topic, 1);//ImagequeryResult


  ros::Rate loop_rate(pub_rate);
  for(int i = 1; i < 6; ++i)
  {  
    std::cout << i <<std::endl;
    if(!nh.ok())  
        break;  
    std::ostringstream stringStream;  
    stringStream << argv[1] << "image" << i << ".jpg";
    std::cout << "stringStream: " << stringStream.str() <<std::endl;
    cv::Mat image = cv::imread(stringStream.str(), CV_LOAD_IMAGE_COLOR);  
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();  
  
    pub.publish(msg);  
    ros::spinOnce();  
    loop_rate.sleep();
    
    if(i==5){
      i=0;
    }
  }  
}

