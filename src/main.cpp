#include <iostream>
#include <fstream>

#include "missionagent.h"
#include "nngkmeans.h"
#include "evaluator.h"

using namespace std;


/*****
@author: Wan-Lei Zhao
@date:   May.-6-2020

This project is an implementation of "On the Merge of k-NN graph"
that is proposed by Wan-Lei Zhao.

***/


void test( )
{
    NNGKMeans::test();
}

int main(int argc, char* argv[])
{
    map<string, const char*> arguments;
    srand(time(0));
    test();
    return 0;
}
