#include "actor_core/ACB.h"
#include "actor_core/actor_types.h"
#include <tinyxml.h>
#include <stdlib.h>
#include "ros/ros.h"
//#include "actor_core/actor_core.h"


void getPlatformResources(std::string xmlContent,std::vector<SensorActuatorInfo>& platformSensors, 
                       std::vector<SensorActuatorInfo>& platformActuators) {

    platformSensors.clear();
    platformActuators.clear();
    //todo replace the function by interpreting the XML file
    //fake a resource 
    ROS_WARN("local resources reading incompleted");
    SensorActuatorInfo sensorInfo1;
    sensorInfo1.id=112;
	sensorInfo1.type="visible";
	sensorInfo1.properties.clear();
    sensorInfo1.params.clear();
    sensorInfo1.params["resolution_x"]="1024";
    sensorInfo1.params["resolution_y"]="768";
    SensorActuatorInfo sensorInfo2;
    sensorInfo2.id=121;
	sensorInfo2.type="distancer";
	sensorInfo2.properties.clear();
    sensorInfo2.params.clear();
    sensorInfo2.properties["resolution"]="0.5";
    platformSensors.push_back(sensorInfo2);
    SensorActuatorInfo actuatorInfo1;
    actuatorInfo1.id=201;
	actuatorInfo1.type="motor";
	actuatorInfo1.properties.clear();
    actuatorInfo1.params.clear();
    platformActuators.push_back(actuatorInfo1);  
}


bool createACB(TaskInfo* pTask,const ResMatchResult* pMatchResult,int64_t actorID,ACB* pACB) {
	
	//pACB->id=???
	pACB->name=pTask->actorName;
	pACB->taskID=pTask->taskID;
	//pACB->pTaskInfo=pTask;
	
	//resources
	std::vector<SensorActuatorInfo> sensors;
	std::vector<SensorActuatorInfo> actuators;
	PlatformInfo* pPlatformInfo;
	SwarmInfo* pSwarmInfo;
	return true;
}
