#ifndef _ACTOR_CALLBACK_QUEUE_H_
#define _ACTOR_CALLBACK_QUEUE_H_
#include <ros/callback_queue.h>
#include <ros/subscription_queue.h>
#include <ros/spinner.h>
#include <string>

// ActorCallbackQueue implement from CallbackQueue,
// this is used for clear callbackQueue when restart the subscriber thread
class ActorCallbackQueue : public ros::CallbackQueue{
    public:
    void clearQueue(){
        while(1){
            uint64_t removalID = -1;
            {
                boost::mutex::scoped_lock lock(mutex_);
                if(callbacks_.empty())break;
                removalID = callbacks_.begin()->removal_id;
                boost::shared_ptr<ros::SubscriptionQueue> qPtr = boost::dynamic_pointer_cast<ros::SubscriptionQueue>(callbacks_.begin()->callback);
                qPtr->clear();
            }
            removeByID(removalID);
        }
    }
};

// every Actor related to one ActorCallbackQueueInfo
// this is used to stop subscribe msgs when pause actor and start subscribe msgs when start actor
class ActorCallbackQueueInfo{
public:
    std::string _actorName;
    boost::shared_ptr<ActorCallbackQueue> _callbackQueue;
    boost::shared_ptr<ros::AsyncSpinner> _asyncSpinner;
    ActorCallbackQueueInfo(std::string actorName):_actorName(actorName){
        _callbackQueue.reset(new ActorCallbackQueue());
        _asyncSpinner.reset(new ros::AsyncSpinner(1, _callbackQueue.get()));
    }
    ~ActorCallbackQueueInfo(){
        _asyncSpinner.reset(); // _asyncSpinner will be auto stoped 
        _callbackQueue.reset();  
    }

    // call this before stop thread
    void stopCallbackQueue(){
        _asyncSpinner->stop();
    }

    // call this before start thread
    void startCallbackQueue(){
        _callbackQueue->clearQueue();
        if(_asyncSpinner->canStart())
            _asyncSpinner->start(); 
    }
};
#endif

