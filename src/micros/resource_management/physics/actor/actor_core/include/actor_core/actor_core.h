#ifndef __ACTOR_CORE__
#define __ACTOR_CORE__

#include<string>
#include<stdio.h>
#include "actor_core/ACB.h"
#include "actor_core/actor_types.h"
#include<time.h>
#include<sys/time.h>
#include<vector>

#define GET_CUR_TIME  \
	timeval timeV;\
	gettimeofday(&timeV,NULL);\
	time_t timeT = timeV.tv_sec;\
	tm *pTm = localtime(&timeT);\
	char szTime[24];\
	sprintf(szTime,"%04d/%02d/%02d %02d:%02d:%02d.%03ld",pTm->tm_year+1900,pTm->tm_mon+1, pTm->tm_mday, pTm->tm_hour, pTm->tm_min, pTm->tm_sec, timeV.tv_usec/1000);



void getPlatformResources(std::string xmlContent,std::vector<SensorActuatorInfo>& platformSensors, std::vector<SensorActuatorInfo>& platformActuators);
bool createACB(TaskInfo* pTask,const ResMatchResult* pMatchResult,int64_t actorID,ACB* pACB);

#endif
