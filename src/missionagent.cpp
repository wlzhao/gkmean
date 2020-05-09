#include "missionagent.h"

#include "abstractkmeans.h"
#include "nngkmeans.h"
#include "evaluator.h"
#include "ioagent.h"
#include "cleaner.h"
#include "vstring.h"

#include <cstring>
#include <string>
#include <iostream>

using namespace std;

bool MissionAgent::buildClust(map<string, const char*>   &arguments)
{
    const char *paras[] = {"-i", "-d", "-k", "-m"};
    char seed[16], crt[16], prr[16];
    unsigned int i = 0;
    bool _refine_  = false;
    for(i = 0; i < 3; i++)
    {
        if(arguments.find(paras[i]) == arguments.end())
        {
            cout<<"Required parameter '"<<paras[i]<<"' is missing!\n";
            return false;
        }
    }
    AbstractKMeans *mykmn = NULL;
    mykmn = new NNGKMeans();
    /**
    if(!strcmp(arguments["-m"], "gkm"))
    {
        mykmn = new NNGKMeans();
    }/**/

    if(arguments.find("-s") != arguments.end())
    {
        strcpy(seed, arguments["-s"]);
    }
    else
    {
        strcpy(seed, "non");
    }
    strcpy(seed, "non");

    if(arguments.find("-r") != arguments.end())
    {
        _refine_ = true;
    }


    if(!strcmp(seed, "rnd") != 0 && !strcmp(seed, "kpp")!= 0 && strcmp(seed, "non") != 0)
    {
        cout<<" Unknown seeding option '"<<seed<<"'!\n";
        cout<<" Valid options are 'rnd', 'kpp' or 'non'!\n";
        exit(0);
    }

    if(arguments.find("-p") != arguments.end())
    {
        strcpy(prr, arguments["-p"]);
    }
    else
    {
        strcpy(prr, "large");
    }

    strcpy(prr, "large");

    if(strcmp(prr, "large") != 0 && strcmp(prr, "best") != 0)
    {
        cout<<" Unknown partition option '"<<prr<<"'!\n";
        cout<<" Valid options are 'large' or 'best'!\n";
        exit(0);
    }

    if(arguments.find("-c") != arguments.end())
    {
        strcpy(crt, arguments["-c"]);
    }
    else
    {
        strcpy(crt, "i2");
    }
    strcpy(crt, "i2");

    int clustNum = atoi(arguments["-k"]);

    if(clustNum < 0 || clustNum > 2147483648)
    {
        cout<<" Invalid cluster number!\n";
        cout<<" Suggested range of cluster number: 1 < k < matrix size\n";
        exit(0);
    }

    mykmn->buildcluster(arguments["-i"], "", seed, prr, crt, clustNum, false);
    mykmn->save_clust(arguments["-d"]);

    delete mykmn;

    return true;
}

bool MissionAgent::buildKNNG(map<string, const char*>  &arguments)
{
    return true;
}
