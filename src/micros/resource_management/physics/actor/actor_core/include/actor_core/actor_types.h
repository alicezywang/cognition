#ifndef __ACTOR_TYPES__
#define __ACTOR_TYPES__
#include<string>
#include<stdint.h>
#include<map>
#include<vector>
#include<string>

class SensorActuatorInfo {
public:
	int64_t id;
	std::string type;
	//static properties of the sensor
	std::map<std::string,std::string> properties;
	//configurable parameters of the sensor
	std::map<std::string,std::string> params;
};

class ResMatchResult {
public:
	//key to resource idx
	std::map<std::string,SensorActuatorInfo> keyToRes;
};

class PluginInfo {
public:
	std::string busName;
	std::string name;
	PluginInfo(const std::string aName,const std::string aBusName):busName(aBusName),name(aName) {}
};

class TaskInfo {
public:
	int64_t taskID;
	std::string actorName;
	std::string taskXMLStr;
	std::vector<PluginInfo> plugins;
};

class SwarmInfo {
public:
	std::string swarmName;
	int64_t swarmID;
	//DB 180601
	int32_t swarmPrio;
	//state
	int16_t state;
	//local platform name
	std::string platformName;
	//actors in the swarm
	std::vector<std::string> generalActors;
	std::map<std::string,std::vector<std::string> > excluActors;
	std::vector<std::string> dynamicActors; 	// [DMY] 20180711
	//platform-->actors in the swarm
	std::map<std::string,std::vector<std::string> > platformActors;

};

class PlatformInfo {
public:
	int64_t id;
	std::string name;
	std::string type;
	std::map<std::string,std::string> properties;	
};



#endif

