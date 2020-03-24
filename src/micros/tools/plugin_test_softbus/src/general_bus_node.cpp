#include <ros/ros.h>
#include <general_bus/general_bus.h>
#include <QApplication>
#include "ros_thread.h"


int main(int argc,char** argv) {
    //sleep(3);
    //QApplication app(argc, argv);
    ros::init(argc,argv,"general_orient_bus");

    // param update
    ros::NodeHandle private_nh("~"); // 私有空间命名
    std::string plugin_name = "orient_softbus::FasterRcnnCognTest";
    private_nh.param("pluginName", plugin_name, plugin_name);

    // Bus
    general_bus::GeneralBus orientBus("orient_bus");
    boost::shared_ptr<ActorCallbackQueue> callbackQueue;

    // Plugin
    orientBus.initPlugin( plugin_name, 1234, callbackQueue); //creat a object from closs_loader
    orientBus.loadPlugin( plugin_name, 1234);             //create thread
    orientBus.startPlugin(plugin_name.c_str(), 1234);    //run


    ros::spin();
    //RosThread ros_thread;
    //app.connect(&ros_thread, SIGNAL(rosShutdown()), &app, SLOT(quit()));
    //app.exec();
    return 0;
}

//rospack plugins --attrib=plugin cognition_bus
