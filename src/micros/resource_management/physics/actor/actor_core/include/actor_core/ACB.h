#ifndef __ACB__
#define __ACB__

#include <stdint.h>
#include <vector>
#include <iterator>
// #include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "actor_core/actor_types.h"
#include "actor_core/actor_constant.h"
#include "actor_callbackqueue.h"
class ACB {
public:
	//actorID and name
	int64_t actorID;
	std::string name;
	int32_t actorNum; // add by gjl ,used by actor request
	
	//task info
	int64_t taskID;
	std::string taskName;
	TaskInfo taskInfo;

	//ACB tree info
	ACB* pParent;
	ACB* pChild;
	ACB* pSibling;

	//actor state
	int16_t state;

	//resources and configurable parameters
	std::vector<SensorActuatorInfo> sensors;
	std::vector<SensorActuatorInfo> actuators;
	
	//resources
	PlatformInfo* pPlatformInfo;
	SwarmInfo* pSwarmInfo;

	//priority
	int32_t prio;	

	//position in a formation
	int8_t formationPos;

	//formation type
	std::string formationType;

	// Set parameters
	void setActorParams(const std::map<std::string, std::string> &aMap){
		if(aMap.empty()) return;
		_paramsMap.insert(aMap.begin(), aMap.end());
	}

	// Get parameters
	void getActorParams(std::map<std::string, std::string> &aMap) const{
		if(!_paramsMap.empty()){
			aMap.insert(_paramsMap.begin(), _paramsMap.end());
		}
	}
private:
	//Actor's parameters
	std::map<std::string, std::string> _paramsMap;
};

class ACBList{
	public:

	//
	bool isEmpty(){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		return _acbList.empty();
	}
	//get the actorID of running actors
	void getRunningActorID(std::vector<int64_t> &aRunningActorIDVtr){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		for(int i=0;i<_acbList.size();i++){
			if(_acbList[i].state==ACTOR_STATE_RUNNING){
				aRunningActorIDVtr.push_back(_acbList[i].actorID);
			}
		}
		return;
	}
	//get the ACB pointer of the running actors
	void getRunningACB(std::vector<ACB*> &aRunningACBVtr){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		for(int i=0;i<_acbList.size();i++){
			if(_acbList[i].state==ACTOR_STATE_RUNNING){
				aRunningACBVtr.push_back(&_acbList[i]);
			}
		}
		return;
	}
	//get the ACBlist iterator of the running
	//get the actorID of the min prio actor
	//if aMinPrioActor = -1, means the acblist is empty
	void getMinPrioActorID(int64_t &aMinPrioActor){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		aMinPrioActor = -1;
		int aMinPrio = MAX_PRIORITY+1;
		for(int i=0;i<_acbList.size();i++){
			if(_acbList[i].prio<aMinPrio){
				aMinPrio = _acbList[i].prio;
				aMinPrioActor = _acbList[i].actorID;
			}
		}
		return;

	}
	//get the pointer of the min prio ACB
	//if the point is NULL, the acblist is empty
	void getMinPrioACB(ACB* &aMinPrioACB){
		aMinPrioACB = NULL;
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		int aMinPrio = MAX_PRIORITY+1;
		for(int i=0;i<_acbList.size();i++){
			if(_acbList[i].prio<aMinPrio){
				aMinPrio = _acbList[i].prio;
				aMinPrioACB = &_acbList[i];
			}
		}
		return;
	}
	//get the iterator of the min prio ACB
	void getMinPrioACB(std::vector<ACB>::iterator &aMinPrioACBItr){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		int minPrio = MAX_PRIORITY+1;
		for(std::vector<ACB>::iterator it=_acbList.begin();it!=_acbList.end();it++){
			if(it->prio<minPrio){
				minPrio = it->prio;
				aMinPrioACBItr = it;
			}
		}
		return;
	}
	//appendACB
	void appendACB(ACB anACB){
		boost::shared_lock<boost::shared_mutex> readLock(_acbListMutex);
		_acbList.push_back(anACB);
		return;
	}

	//shared mutex for acblist
	boost::shared_mutex _acbListMutex; 

	private:

	std::vector<ACB> _acbList;


};

//actor information for plugin
struct ActorPluginInfo{
	int64_t _actorID;
	std::string _actorName;
	int16_t _state;
	TaskInfo _taskInfo;
	//Actor's parameters
	std::map<std::string, std::string> _paramsMap;

	// this is an actor callbackQueue,shared by its plugins
	boost::shared_ptr<ActorCallbackQueue> _callbackQueue;
	~ActorPluginInfo(){
		if(_callbackQueue)
			_callbackQueue.reset();
	}
};
#endif

