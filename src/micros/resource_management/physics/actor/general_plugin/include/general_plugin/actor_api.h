#ifndef __ACTOR_API__
#define __ACTOR_API__


#include <string>
#include "boost/shared_ptr.hpp"
#include "actor_core/barrier.h"
extern int SHARED_DATA_SIZE;

/*class declration*/
class ActorScheduler;
class SwarmMasterCmd;
namespace general_bus {
    class GeneralPlugin;
}

/*global variable and functions*/
extern  ActorScheduler *gPActorScheduler;
extern SwarmMasterCmd *gSwarmMasterCmd;
//get actor name by actorID
//extern bool getActorName(int64_t anActorID,std::string &anActorName);
//get plugin pointer by pluginname 
extern bool getActorPlugin(int64_t anActorID,std::string aPluginName,boost::shared_ptr<general_bus::GeneralPlugin> &aPluginStr);
//get and set shareddata
extern char* gPSharedData;
extern int getSharedData(char* aPData,int aSize);
extern int setSharedData(char* aPData,int aSize);
//switch actor
extern void switchToActor(int64_t anActorID,std::string anActorName);

//activate actor
extern void activateActor(int64_t anActorID,std::string anActorName);
//pause actor
extern void pauseActorApi(int64_t anActorID);
//quit actor
extern void appendScheduleRequest(int64_t anActorID);
//get formation info by actorID
extern bool getFormation(int64_t anActorID, std::string &aFormationType, std::string &aFormationPos);

// request actor from master
 void requestSwarmActorFromMaster(const std::string& anActorName);
 void requestActivateActorFromMaster(const std::string& aSendActorName,const std::string& anActorName);
 void requestSwitchActorFromMaster(const std::string& aSendActorName,const std::string& anActorName);

// robot barrier api, for all robot step into next state in the same time
 BarrierResult pluginBarrierApi(GlobalBarrierKey& aBarrierKey, int aWaitCount, const short aTimeout);

#endif
