#ifndef TRAIN_PLUGIN_H
#define TRAIN_PLUGIN_H

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

namespace orient_softbus{
using namespace std;

class FasterRcnnCognTrain: public general_bus::GeneralPlugin
{
public:
    FasterRcnnCognTrain();
    ~FasterRcnnCognTrain();
    virtual void start();

private:
    ros::NodeHandle nh;
    ros::Subscriber sub_img;
    ros::Publisher  pub_msg;
    ros::Publisher  pub_test;
    int iter;
    int max_iters;
    string output_dir;

    boost::shared_ptr<cognition::Handle> congnition_softbus;

};//class

}//namespace


#endif // TRAIN_H
