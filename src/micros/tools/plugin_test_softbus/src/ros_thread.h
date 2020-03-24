#ifndef ROS_THREAD_H
#define ROS_THREAD_H
#include <ros/ros.h>
#include <boost/thread.hpp>
#include <QObject>

class RosThread : public QObject
{
    Q_OBJECT

public:
    RosThread(QObject *parent = 0);
    ~RosThread();
    void rosrunThread();

signals:
    /** @brief Emitted when ros Shutdown. */
    void rosShutdown();

private:
    bool m_isExt;
    boost::thread *m_ros_thread;
};
#endif // ROS_THREAD_H
