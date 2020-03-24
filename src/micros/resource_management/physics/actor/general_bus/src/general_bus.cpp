#include "general_bus/general_bus.h"
#include "ros/ros.h"
#include <pluginlib/class_loader.h>
#include <string>

const int gRealTime_CPU=0; 
namespace general_bus {
	GeneralBus::GeneralBus(const char* busName):_pluginLoader("general_plugin","general_bus::GeneralPlugin"),_nRegisteredPlugin(0) {
		_name = std::string(busName);
		_nRegisteredPlugin = 0;
		//_plugin2ActorID.clear();
	}

	GeneralBus::~GeneralBus() {
		//todo list:
		//stop plugins
		for(int i=0; i<_nRegisteredPlugin; i++){
			_pluginList[i]->stop();
			_pluginList[i].reset();
		}
		//unload plugins
		_nRegisteredPlugin = 0;
	}

	//Initialize plugin before loading
	bool GeneralBus::initPlugin(const std::string aPluginName, int64_t anActorID, boost::shared_ptr<ActorCallbackQueue> callbackQueue){
		assert(_nRegisteredPlugin < PLUGIN_NUM_MAX-1);
		boost::unique_lock<boost::shared_mutex> writeLock(_pluginListMutex);
		try {
			_pluginList[_nRegisteredPlugin] = _pluginLoader.createInstance(aPluginName.c_str());
			_pluginOwner[_nRegisteredPlugin] = anActorID;
			_pluginName[_nRegisteredPlugin] = aPluginName;
			ROS_INFO("[General Bus] Actor %ld is initialize plugin %s on %s", anActorID, aPluginName.c_str(), _name.c_str());
        }
		catch (const pluginlib::PluginlibException& ex) {
			ROS_FATAL("[General Bus] Failed to create the %s device, are you sure it is properly registered and that the containing library is built? Exception: %s", aPluginName.c_str(), ex.what());
			return false;
        }
		_pluginList[_nRegisteredPlugin]->initialize(anActorID, aPluginName, callbackQueue);
        _nRegisteredPlugin++;
		return true;
    }

	//Load a plugin in a new thread
	void GeneralBus::loadPlugin(const std::string aPluginName, int64_t anActorID){
		boost::unique_lock<boost::shared_mutex> lock(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if(_pluginOwner[i] == anActorID && _pluginName[i] == aPluginName){
				//create the thread
				boost::thread* newThread = _pluginTG.create_thread(boost::bind(&GeneralPlugin::run, _pluginList[i]));
				// if pluginName start with rt, then setaffinity 
				if(strstr(aPluginName.c_str(), "rt") == aPluginName.c_str()){
					setAffinity(newThread, gRealTime_CPU);
				}
				ROS_INFO("[General Bus] Actor %ld is loading plugin %s on %s", anActorID, aPluginName.c_str(), _name.c_str());
				break;
            }
        }
	}

	std::string GeneralBus::getBusName()
	{
		return _name;
	}

	bool GeneralBus::startActorPlugins(int64_t anActorID){
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if(anActorID == _pluginOwner[i]){
				_pluginList[i]->resume();
			}
		}
		return true;
	}
	
	bool GeneralBus::stopActorPlugins(int64_t anActorID){
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if(anActorID == _pluginOwner[i]){
				_pluginList[i]->stop();
			}
		}
		return true;
	}

	bool GeneralBus::pauseActorPlugins(int64_t anActorID){
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if(anActorID == _pluginOwner[i]){
				ROS_INFO("[General Bus] test: pauseing plugin %s in actor %ld ", _pluginName[i].c_str(), _pluginOwner[i]);
				_pluginList[i]->pause();
				ROS_INFO("[General Bus] test: pause plugin %s in actor %ld ", _pluginName[i].c_str(), _pluginOwner[i]);
			}
		}
		ROS_INFO("[General Bus] test: pauseActorPlugins return..");
		return true;
	}

	bool GeneralBus::updateActorPluginState(ActorPluginInfo &anActor,const int anState){
		//may need to be parallel
		bool setResult = true;
		switch (anState){
			case ACTOR_STATE_INIT_DELAY:
			case ACTOR_STATE_INIT:
				for(int i=0; i<anActor._taskInfo.plugins.size(); i++){
					std::string tBusName = anActor._taskInfo.plugins[i].busName;
					if(tBusName.compare(_name) == 0){
						setResult = setResult && initPlugin(anActor._taskInfo.plugins[i].name, anActor._actorID, anActor._callbackQueue);
						//If set fails return
						if(!setResult) break;
						//Set plugin's parameters between initialization and loading
						setPluginParams(anActor._taskInfo.plugins[i].name, anActor._actorID, anActor._paramsMap);
						loadPlugin(anActor._taskInfo.plugins[i].name, anActor._actorID);
					}
				}
				break;
			case ACTOR_STATE_RUNNING:
				//if the actor's state is INIT, init and load plugins first
				if(anActor._state == ACTOR_STATE_INIT || anActor._state == ACTOR_STATE_INIT_DELAY){
					setResult = updateActorPluginState(anActor,ACTOR_STATE_INIT);
				} else {
					for(int i=0; i<anActor._taskInfo.plugins.size(); i++){
						std::string tBusName = anActor._taskInfo.plugins[i].busName;
						if(tBusName.compare(_name) == 0){
							setPluginParams(anActor._taskInfo.plugins[i].name, anActor._actorID, anActor._paramsMap);
						}
					}
				}
				setResult = setResult && startActorPlugins(anActor._actorID);
				break;
			case ACTOR_STATE_PAUSE:
				setResult = pauseActorPlugins(anActor._actorID);
				break;
			case ACTOR_STATE_FINISH:
				setResult = stopActorPlugins(anActor._actorID);
				break;
			default:
				setResult = false;			
		}
		return setResult;
	}

	bool GeneralBus::getActorPlugin(int64_t anActorID, std::string aPluginName, boost::shared_ptr<GeneralPlugin> &aPluginStr) {
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if((_pluginOwner[i] == anActorID) && (_pluginName[i].compare(aPluginName) == 0)){
				aPluginStr = _pluginList[i];
				return true;
			}
		}
		return false;
	}

	bool GeneralBus::setAffinity(boost::thread* th, int cpu){
		std::stringstream ss;
		unsigned long long threadID = 0;
		ss<<th->native_handle();
		ss>>threadID;

		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(cpu, &mask);
		ROS_INFO("[General Bus] set thread[%llu] on cpu %d", threadID, cpu);
		if(pthread_setaffinity_np(threadID, sizeof(mask), &mask) != 0)
		ROS_ERROR("[General Bus] Failed to set thread[%llu] on cpu %d.", threadID, cpu);
	}

	void GeneralBus::setPluginParams(const std::string aPluginName, int64_t anActorID, const std::map<std::string, std::string> &aMap){
		boost::shared_lock<boost::shared_mutex> read(_pluginListMutex);
		for(int i=0; i<_nRegisteredPlugin; i++){
			if(_pluginOwner[i] == anActorID && _pluginName[i] == aPluginName){
				_pluginList[i]->setParams(aMap);
				break;
			}
		}
    }

	/*Unused*/
	int GeneralBus::getActorPluginState(const char* pluginName, int64_t actorID){
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		std::string name = std::string(pluginName);
		for(int k=0; k<_nRegisteredPlugin; k++){
			if((_pluginName[k].compare(name) == 0) && (_pluginOwner[k] == actorID)){
				return _pluginList[k]->getState();
			}
		}
		return -1;
	}

	/*Deprecated*/
	bool GeneralBus::startPlugin(const char* pluginName,int64_t actorID) {
		std::string tempName = std::string(pluginName);
		int i = 0;
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(;i<_nRegisteredPlugin;i++){
			if((tempName.compare(_pluginList[i]->getName()) == 0) && (actorID == _pluginList[i]->getActorID())){
				_pluginList[i]->resume();
				break;
			}
		}
		if(i == _nRegisteredPlugin){
			ROS_WARN("[General Bus]Plugin %s not loaded",pluginName);
			return false;
		}

		ROS_INFO("[General Bus]Actor %ld is starting plugin %s on %s",actorID,pluginName,_name.c_str());
		return true;
	}
  
    /*Deprecated*/
    bool GeneralBus::pausePlugin(const char* pluginName, int64_t actorID) {
		std::string tempName = std::string(pluginName);
		int i = 0;
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(; i<_nRegisteredPlugin; i++){
			if((tempName.compare(_pluginList[i]->getName()) == 0) && (actorID == _pluginList[i]->getActorID())){
				_pluginList[i]->pause();
				break;
			}
		}
		if(i == _nRegisteredPlugin){
			ROS_WARN("[General Bus]Plugin %s not loaded",pluginName);
			return false;
		}
		ROS_INFO("[General Bus]Actor %ld is pausing plugin %s on %s", actorID,pluginName, _name.c_str());
		return true;		
	}

    /*Deprecated*/
    bool GeneralBus::stopPlugin(const char* pluginName, int64_t actorID) {
		std::string tempName = std::string(pluginName);
		int i = 0;
		boost::shared_lock<boost::shared_mutex> readLock(_pluginListMutex);
		for(; i<_nRegisteredPlugin; i++){
			if((tempName.compare(_pluginList[i]->getName()) == 0) && (actorID == _pluginList[i]->getActorID())){
				_pluginList[i]->stop();
				break;
			}
		}
		if(i == _nRegisteredPlugin){
			ROS_WARN("[General Bus]Plugin %s not loaded",pluginName);
			return false;
		}
		ROS_INFO("[General Bus]Actor %ld is stoping plugin %s on %s",actorID,pluginName,_name.c_str());
		return true;
	}

	/*Deprecated*/
	bool GeneralBus::unloadPlugin(const char* pluginName,int64_t actorID) {
		ROS_FATAL("[General Bus]GeneralBus::unloadPlugin not implemented");
	}

}

