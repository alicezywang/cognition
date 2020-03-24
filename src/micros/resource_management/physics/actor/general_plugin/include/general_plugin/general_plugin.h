#ifndef __GENERAL_PLUGIN__
#define __GENERAL_PLUGIN__

#include<stdio.h>
#include<boost/thread/condition.hpp>
#include<boost/thread/mutex.hpp>
#include<boost/thread/detail/thread.hpp>
#include<boost/thread/thread_time.hpp>
#include<boost/function.hpp>
#include<boost/bind.hpp>
#include<boost/thread/shared_mutex.hpp>
#include<string>
#include"ros/ros.h"
#include"actor_core/actor_constant.h"
#include "actor_api.h"
#include "actor_core/actor_callbackqueue.h"
#include <exception>  
namespace general_bus {
	class GeneralPlugin {
	public:
		virtual ~GeneralPlugin(){
			if(_callbackQueue)
				_callbackQueue.reset();
		}
		std::string getName(){
			return _name;
		}

		int64_t getActorID(){
			return _actorID;
		}

		bool initialize(int64_t aActorID,std::string aPluginName,boost::shared_ptr<ActorCallbackQueue> aCallbackQueue=boost::shared_ptr<ActorCallbackQueue>());

		int getState() {
			boost::mutex::scoped_lock lock(_mutex);
			return _state;
		}
	
		void setState(int aState) {
			_state=aState;
		}

		boost::condition_variable* getCond() {
			return &_cond;
		}
		
		boost::mutex* getMutex() {
			return &_mutex;
		}

		//for scheduler to pause this actor, thread safe required
		void pause();
		//for scheduler to wakeup this actor, thread safe required
		void resume();
		//for scheduler to stop this actor, thread safe required
		void stop();
		//for the actor to terminate itself
		//if you want to notify the schedueler to re-schedule, call terminate(),else GOON
		void terminate()  {
			{
				boost::mutex::scoped_lock lock(_mutex);
				setState(ACTOR_STATE_FINISH); //use constants like ACTOR_STATE_STOP;
				_cond_pause.notify_one();
				//the same as _cond_pause
				_cond_stop.notify_one();
			}
			//notify the schedule
			ROS_INFO("[General Plugin] Actor actor ID %ld ,plugin %s call terminate, before scheduler lock",_actorID,_name.c_str());
			/*
			if((_pMutex==NULL)||(!_pMutex)){
				ROS_ERROR("[General Plugin] pMutex = NULL");
			}
			boost::mutex::scoped_lock scheLock(*_pMutex);
			_pCond->notify_one(); */
			appendScheduleRequest(_actorID);
			
			ROS_INFO("[General Plugin] Actor %ld , plugin %s Terminated",_actorID,_name.c_str());
			pthread_exit(0);	
		}  

		//yield, for the actor to pause itself
		void yieldActor(){
			{
				boost::mutex::scoped_lock lock(_mutex);
				setState(ACTOR_STATE_PAUSE);
				_cond_pause.notify_one();
			}
			pauseActorApi(_actorID);
			//GOON_OR_RETURN,the plugin would stop this interrupt point
			insertInterruptablePoint();
		}
		
		//insert interrupt point, GOON_OR_RETURN
		void insertInterruptablePoint();

		#define GOON_OR_RETURN  insertInterruptablePoint();
		/* {	boost::mutex::scoped_lock lock(_mutex);\
			if (_state==ACTOR_STATE_PAUSE)  {  \
				_cond_pause.notify_one();\
				_cond.wait(lock);\
			}\
			if(_state==ACTOR_STATE_FINISH){\
				_cond_stop.notify_one();\
				}\
				if(_state==ACTOR_STATE_FINISH) pthread_exit(0) ;\
		}\ */
			
		//for test service 
		#define SERVICE_CALLBACK_INIT if(_state!=ACTOR_STATE_RUNNING) return false;
		#define MESSAGE_CALLBACK_INIT if(_state!=ACTOR_STATE_RUNNING) return ;

		//overload start()
		virtual void start() {}	

		// This method will be called to transmit params from UTO to plugins before plugins starting
		void setParams(const std::map<std::string, std::string> &aMap);
		// Get param
		std::string getParam(const std::string &paramName);
		// TODO...Uto call this method to change plugins' params after they starting
		virtual void setParamsRuntime() {}

		//create thread
		void run(){
			insertInterruptablePoint();
			try{
				start();
			}catch(std::exception& e){
				ROS_ERROR("[General Plugin] Actor plugin %s exit with error msg:%s!", typeid(this).name(), e.what());
			}catch(...){
				ROS_ERROR("[General Plugin] Actor plugin %s exit with error!", typeid(this).name());
			}
			
			while(true) {
				GOON_OR_RETURN;
				usleep(1000);
			}
		}
		
		int _state;
		int _duration;

		boost::condition_variable _cond; //condition to wakeup this actor
		boost::mutex _mutex;	//mutex to wakeup this actor and protect the _state
		
		boost::condition_variable _cond_pause;
		boost::condition_variable _cond_stop;

		/* boost::condition_variable* _pCond; //condition to wakeup the scheduler
		boost::mutex* _pMutex; //mutex to wakeup the sceduler */
	 
		//std::string _actorName;
		int64_t _actorID;
		std::string _name;

		// Pub event msg
		ros::Publisher _pub;

		//depecated
		boost::shared_ptr<ActorCallbackQueue> getCallbackQueue(){return _callbackQueue;}

		// when async threads create ros pub/sub at the same time, the result is incorrect, such as pub/sub matched twice.
		// to avoid this bug in ros, we reimplement these method, and lock it when create pub/sub.
		// besides, in subscribe methods, we bind a callbackqueue to these created subscribers.
		// call this method when create ros publisher
		template <class M>
    	ros::Publisher pluginAdvertise(const std::string& topic, uint32_t queue_size, bool latch = false){
			ros::AdvertiseOptions ops;
			ops.template init<M>(topic, queue_size);
			ops.latch = latch;
			boost::unique_lock<boost::mutex> lock(_rosCreatePubSubMutex);
			return _handle.advertise(ops);
		}

		// call this method when create ros subscriber
		template<class M, class T>
		ros::Subscriber pluginSubscribe(const std::string& topic, uint32_t queue_size, void(T::*fp)(const boost::shared_ptr<M const>&), T* obj, const ros::TransportHints& transport_hints = ros::TransportHints()){
			ros::SubscribeOptions ops;
			ops.template init<M>(topic, queue_size, boost::bind(fp, obj, _1));
			ops.transport_hints = transport_hints;
			ops.callback_queue = _callbackQueue.get();
			boost::unique_lock<boost::mutex> lock(_rosCreatePubSubMutex);
			return _handle.subscribe(ops);
		}

		// call this method when need sleep in plugin
		void pluginUSleep(const int aUSeconds);

		// Publish event msg to actor state machine
		void pubEventMsg(const std::string &anEventName);
	private:
		boost::shared_ptr<ActorCallbackQueue> _callbackQueue;
		static boost::mutex _rosCreatePubSubMutex;
		ros::NodeHandle _handle;
		std::map<std::string, std::string> _paramsMap;
	};
}

#endif
