
#include <string>
#include <vector>
#include <map>
// task.xml
struct SensorActuatorDesc {
    int64_t _id;
    std::string _type;
    //static properties of the sensor
    std::map<std::string, std::string> properties;
    //configurable parameters of the sensor
    std::map<std::string, std::string> params;
};


struct SwarmActorDesc{
    std::string _actorName;
    int64_t _actorID;
	int32_t _actorNum;
    int32_t _actorPrio;	

    std::map<std::string, std::vector<std::string> > _plugins; // <busname, <pluginname> >

    std::vector<SensorActuatorDesc> _sensors;
    std::vector<SensorActuatorDesc> _actuators; //

    std::map<std::string, std::string> _params;     // uto, <param, paramvalue>
    std::map<std::string, std::string> _transition; // uto, <condition, next actor>
};

struct SwarmTaskDesc{
    std::string _taskName;
    int64_t _taskID;
    int32_t _taskPrio;
	int16_t _taskState;
    std::vector<SwarmActorDesc> _taskActors;
};