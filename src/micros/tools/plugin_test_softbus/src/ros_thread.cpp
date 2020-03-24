#include "ros_thread.h"
#include <iostream>

RosThread::RosThread(QObject *parent)
    : QObject(parent)
{
//    if (!ros::master::check())
//    {
//        return false;
//    }
    ros::start(); // explicitly needed since our nodehandle is going out of scope.

    m_isExt = false;
    m_ros_thread = NULL;
    m_ros_thread = new boost::thread(boost::bind(&RosThread::rosrunThread, this));
}

RosThread::~RosThread()
{
    m_isExt = true;
    if(m_ros_thread)
    {
        m_ros_thread->interrupt();
        m_ros_thread->join();
        delete m_ros_thread;
    }
    std::cout << "Ros Thread Closed." << std::endl;

}

void RosThread::rosrunThread()
{
    ros::Duration initDur(0.2);
    while (ros::ok() && !m_isExt)
    {
        ros::spinOnce();
        initDur.sleep();
    }

    std::cout << "Ros shutdown, proceeding to close the Gui ..." << std::endl;
    Q_EMIT rosShutdown();
}