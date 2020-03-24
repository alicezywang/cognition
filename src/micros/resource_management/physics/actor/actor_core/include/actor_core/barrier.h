#ifndef __BARRIER_H__
#define __BARRIER_H__
#include <string>

enum BarrierResult{NO=0, YES, OPTIONALYES, TIMEOUT};

struct GlobalBarrierKey_{
    std::string _swarmName; 
    std::string _actorName;
    std::string _pluginName ;   // this param is invalid in actor barrier process
    short _barrierKey;          // unique key

    GlobalBarrierKey_(const std::string& anActorName, const short aKey)
        :_swarmName(""), _actorName(anActorName), _pluginName(""), _barrierKey(aKey){}

    GlobalBarrierKey_(const std::string& anActorName, const std::string& aPluginName, const short aKey)
        :_swarmName(""), _actorName(anActorName), _pluginName(aPluginName), _barrierKey(aKey){}

    GlobalBarrierKey_(const std::string& aSwarmName, const std::string& anActorName, const std::string& aPluginName, const short aKey)
        :_swarmName(aSwarmName), _actorName(anActorName), _pluginName(aPluginName), _barrierKey(aKey){}
    
    friend bool operator <(const GlobalBarrierKey_& key1, const GlobalBarrierKey_& key2){
        if(key1._swarmName < key2._swarmName) return true;
        if(key1._swarmName == key2._swarmName){
            if(key1._actorName < key2._actorName) return true;
            if(key1._actorName == key2._actorName){
                if(key1._pluginName < key2._pluginName) return true;
                if(key1._pluginName == key2._pluginName) return key1._barrierKey < key2._barrierKey;
            }
        }
        return false;
    }
};
// 
struct GlobalBarrierKey{
    short _barrierKey;          // unique key
    GlobalBarrierKey(const short aKey):_barrierKey(aKey){}
    friend bool operator <(const GlobalBarrierKey& key1, const GlobalBarrierKey& key2){
        return  key1._barrierKey < key2._barrierKey;
    }
    //depecated 
    GlobalBarrierKey(const std::string& anActorName, const short aKey):_barrierKey(aKey){}

    GlobalBarrierKey(const std::string& anActorName, const std::string& aPluginName, const short aKey):_barrierKey(aKey){}

    GlobalBarrierKey(const std::string& aSwarmName, const std::string& anActorName, const std::string& aPluginName, const short aKey)
        :_barrierKey(aKey){}
};

#endif