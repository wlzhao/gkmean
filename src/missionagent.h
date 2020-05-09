#ifndef MISSIONAGENT_H
#define MISSIONAGENT_H

/**
* in charge of distributing the tasks
*
*
*@author:  Wan-Lei Zhao
*@date:    Feb.-23-2017
*
*
**/

#include <cstring>
#include <string>
#include <map>

using namespace std;

class MissionAgent
{
    public:
        MissionAgent(){}
        virtual ~MissionAgent(){}
        static bool buildClust(map<string,  const char*> &arguments);
        static bool buildKNNG(map<string, const char*>  &arguments);

};

#endif
