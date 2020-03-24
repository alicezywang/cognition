#include "general_plugin/general_plugin.h"
#include "actor_msgs/UTOEvent.h"

namespace general_bus {
	boost::mutex GeneralPlugin::_rosCreatePubSubMutex;
	bool GeneralPlugin::initialize(int64_t aActorID,std::string aPluginName,boost::shared_ptr<ActorCallbackQueue> aCallbackQueue){
		//_actorName=aActorName;
		_actorID = aActorID;
		_name = aPluginName;
		_state = ACTOR_STATE_PAUSE;
		_duration = 1;
		_callbackQueue = aCallbackQueue;
		// Publish event msg to actor state machine
		_pub = pluginAdvertise<actor_msgs::UTOEvent>("uto_event_msg", 10000);
	}

	void GeneralPlugin::pause() {
		boost::mutex::scoped_lock lock(_mutex);
		if(_state==ACTOR_STATE_FINISH){
			ROS_INFO("[General Plugin] Actor %ld, plugin %s Stoped",_actorID, _name.c_str());
			return;
		}
		if(_state==ACTOR_STATE_PAUSE){
			ROS_INFO("[General Plugin] Actor %ld, plugin %s Paused",_actorID,_name.c_str());
			return;

		}
		setState(ACTOR_STATE_PAUSE);
		_cond_pause.wait(lock);
		ROS_INFO("[General Plugin] Actor %ld, plugin %s Paused",_actorID,_name.c_str());
	}

	void GeneralPlugin::resume() {
		boost::mutex::scoped_lock lock(_mutex);	
		setState(ACTOR_STATE_RUNNING);
		_cond.notify_one();
	}

	void GeneralPlugin::stop(){
		boost::mutex::scoped_lock lock(_mutex);
		if(_state==ACTOR_STATE_FINISH){
			ROS_INFO("[General Plugin] Actor %ld, plugin %s Stoped",_actorID,_name.c_str());
			return;
		}
		setState(ACTOR_STATE_FINISH);
		_cond_stop.wait(lock);
		ROS_INFO("[General Plugin] Actor %ld, plugin %s Stoped",_actorID,_name.c_str());
	}

	/*  void GeneralPlugin::terminate() {
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
		/*
		appendScheduleRequest(_actorID);
		
		ROS_INFO("[General Plugin] Actor %ld , plugin %s Terminated",_actorID,_name.c_str());
		pthread_exit(0);	
	}  
	*/

	void GeneralPlugin::insertInterruptablePoint(){
		boost::mutex::scoped_lock lock(_mutex);
		if (_state==ACTOR_STATE_PAUSE) { 
			_cond_pause.notify_one();
			_cond.wait(lock);
		}
		if(_state==ACTOR_STATE_FINISH){
			_cond_stop.notify_one();
			pthread_exit(0);
		}
		
	}

	void GeneralPlugin::pubEventMsg(const std::string &anEventName){
		std::string actor;
    //getActorName(_actorID, actor);
		actor_msgs::UTOEvent event;
		event.type = 2; //EVENT_MSG
		event.currentActor = actor;
		event.eventName = anEventName;
		_pub.publish(event);
	}
	
	// This method will be called to transmit params to plugins before plugins starting
	void GeneralPlugin::setParams(const std::map<std::string, std::string> &aMap) {
		if(aMap.empty()) return;
		_paramsMap.insert(aMap.begin(), aMap.end());
	}

	std::string GeneralPlugin::getParam(const std::string &paramName){
		if(paramName.empty() || _paramsMap.count(paramName)==0) return "";
		return _paramsMap[paramName];
	}

	void GeneralPlugin::pluginUSleep(const int aUSeconds){
		int loopCount = aUSeconds / 100;
		int leftUSeconds =  aUSeconds % 100; 
		if(leftUSeconds > 0){
			usleep(leftUSeconds);
		}
		while(loopCount > 0){
			GOON_OR_RETURN;
			usleep(100);
			loopCount--;
		}
	}
}

