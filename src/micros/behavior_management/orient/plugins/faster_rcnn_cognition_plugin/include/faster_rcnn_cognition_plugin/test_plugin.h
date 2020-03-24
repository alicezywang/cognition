#ifndef TEST_PLUGIN_H
#define TEST_PLUGIN_H

// C and C++ headers
#include <iostream>

// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <orient_softbus_msgs/RecogResult.h>
#include <general_plugin/general_plugin.h>

// Interface headers
#include <cognition/cognition.h>

/**
 * @brief namespace orient_softbus
 */
namespace orient_softbus {
using namespace std;
using namespace cognition;

/**
 * @brief The FasterRcnnCognTest class
 */
class FasterRcnnCognTest: public general_bus::GeneralPlugin
{
public:
    FasterRcnnCognTest();
    ~FasterRcnnCognTest();
    virtual void start();

private:
    ros::NodeHandle nh;
    ros::Subscriber sub_img;
    ros::Publisher  pub_msg;
    ros::Publisher  pub_test;
    boost::shared_ptr<cognition::Handle> cognition_handle;

    void imgCallBack(const sensor_msgs::Image::ConstPtr &msg);
};//class

}//namespace


#endif // TEST_PLUGIN_H
