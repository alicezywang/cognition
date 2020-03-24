#ifndef __GENERAL_BUS__
#define __GENERAL_BUS__

#define PLUGIN_NUM_MAX 128

#include "general_plugin/general_plugin.h"
#include <stdint.h>
#include <vector>
#include <map>
#include <pluginlib/class_loader.h>
#include "actor_core/ACB.h"
#include "actor_core/actor_callbackqueue.h"
namespace general_bus {

	class GeneralBus {
	public:
		GeneralBus(const char* busName);
		~GeneralBus();
		//Load a plugin in a new thread
		void loadPlugin(const std::string aPluginName, int64_t anActorID);
		//Initialize plugin before loading
		bool initPlugin(const std::string aPluginName, int64_t anActorID, boost::shared_ptr<ActorCallbackQueue> callbackQueue);

		//set actor plugins
		bool startActorPlugins(int64_t anActorID);
		bool pauseActorPlugins(int64_t anActorID);
		bool stopActorPlugins(int64_t anActorID);

		//set state of plugins of the actor identified by anActorID
		bool updateActorPluginState(ActorPluginInfo &anActor,const int anState);

		//boost::condition_variable* _pCond; //condition to wakeup the scheduler
		//boost::mutex* _pMutex; //mutex to wakeup the sceduler
			
		std::string getBusName();
		int32_t getPluginNum(){
			return _nRegisteredPlugin;
		}

		//get actor plugin shared_ptr by actorID and pluginname
		//require plugin has exclusive name
		bool getActorPlugin(int64_t anActorID,std::string aPluginName,boost::shared_ptr<GeneralPlugin> &aPluginStr);
		//thread group contains plugin thread, one thread for one plugin
		boost::thread_group _pluginTG;
		//shared mutex for pluginlist
		boost::shared_mutex _pluginListMutex;

		/*Unused*/
		int getActorPluginState(const char* pluginName,int64_t actorID);

		/*Deprecated*/
		bool unloadPlugin(const char* pluginName,int64_t actorID);
		bool startPlugin(const char* pluginName,int64_t actorID);
		bool pausePlugin(const char* pluginName,int64_t actorID);
		bool stopPlugin(const char* pluginName,int64_t actorID);

	private:
		// set affinity to cpu
		bool setAffinity(boost::thread* th, int cpu=0);

		pluginlib::ClassLoader<GeneralPlugin> _pluginLoader;
		int64_t _pluginOwner[PLUGIN_NUM_MAX];
		std::string _pluginName[PLUGIN_NUM_MAX];
		int32_t _nRegisteredPlugin;
		boost::shared_ptr<GeneralPlugin>  _pluginList[PLUGIN_NUM_MAX];
		std::string _name;

		//Set plugin's parameters
		void setPluginParams(const std::string aPluginName, int64_t anActorID, const std::map<std::string, std::string> &aMap);
	};

};
#endif
