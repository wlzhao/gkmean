/**
    @function: kmeans based on neiborhood
    @author: chenghao deng
    @version: 2.0
    @date: 2016-2-27
    @institute: Xiamen University


**/
#ifndef NNGKMeans_H
#define NNGKMeans_H

#include "abstractkmeans.h"
#include <set>
#include <unordered_set>

#include "evaluator.h"

class NNGKMeans: public AbstractKMeans
{
    const static unsigned NBIter, TreeNum;
    const static unsigned NIter;
    const static unsigned RangeFirst;
    const static unsigned RangeSecond;
    const static float Err0;

    bool outNN = false;

    unsigned long longloc;///long compute;

    double *Es, *arrayD;
    int     *Ns;
    double *Ds;
    unsigned *topList;
    ///write by pclin 2018-2-27
    unsigned int*rkNNNum    = NULL;
    unsigned int**rkNNGraph = NULL;
    float *distlist = (float*)calloc(40,sizeof(float));

    float *topDist;

    unordered_set<unsigned> *visitList;
    vector<unsigned int> idx2ids;

    float *radius;
    unsigned char *prUpdated, *crUpdated;

    const static unsigned long Treesize;
    const static unsigned long TopK;
    const static unsigned long NNTop;

    int adjustNN();
    int optzI2(bool verbose, const unsigned NNTop,  const unsigned niter0);
    int NewOptzI2(bool verbose, const unsigned NNTop,  const unsigned niter0);
    int initNNGraph();
    int buildNNGraphBy2Mn(int verbose);
    int updateNNGraph();
    int buildNNGraph4clust();

public :
    ///-----for temp used-------///
    float *res;
    unsigned num;
    double tmptime;
    double *tmLogs;
    double *dstLogs;
    unsigned int nLogs;
    ///--------end-------------///
    NNGKMeans();
    virtual ~NNGKMeans();
    void freeOperation();
    bool   init(const char *srcfn);
    bool   init(unsigned char *mat, const int row, const int dim);
    bool   init(float *mat, const int row, const int dim);
    bool   setLogOn(const unsigned logSize);

    int    recall();
    int    NNI2(bool verbose, const unsigned niter0); //gkmeans
    int    buildCluster(unsigned* NN, float* data, unsigned row, unsigned ndim, unsigned clustnumb);
    int    buildCluster(const char *srcfn, const char *nnfn, const char *dstfn, unsigned clustnumb);

    int    copy2TopList(unsigned *top);
    int    copyFromTopList(unsigned *top, unsigned topNum);
    bool   config(const char *_seed_, const char *lg_first, const char *crtrn, int verbose);
    float  l2(const unsigned int i, const unsigned int j);

    int    clust(const unsigned int clust_num, const char *dstfn, const int verbose); //gkmeans interface

    int    buildKNNGraph(const char *srcfn, const char *dstfn, const int dpg);
    void   diversify_by_cut(unsigned int *kNNGraph, const unsigned int N, const unsigned int K, const int edge_num);
    void   augRvsKNN(unsigned int *kNNGraph, unsigned long N, unsigned long D,unsigned int L);


    bool   rndSeeds(const int k, int rseeds[], const int bound);
    bool   kppSeeds(const int k, int rseeds[], const int bound);

    void   saveCenters(const char *dstfn,  bool append);
    void   saveKNNGraph(const char *dstfn, unsigned int k0);
    unsigned  *loadKNNGraph(const char *srcfn);
    void   saveKNNGraph(const char *dstfn, const unsigned int n, const unsigned int k0, vector<unsigned int> &ids);
    void   saveKNNGraphDPG(const char *dstfn, unsigned int k0);
    void   saveKNNGraphDPG(const char *dstfn, const unsigned int N, const unsigned int D,
                           unsigned int **knnGraph, unsigned int *knnNumb);
    int    fetchCenters(float *centers);
    ///write by pclin
    void   saveKGraph(const char *dstfn,unsigned int tpk);
    void   saveLogs(const char *dstFn);

    static void test();
    static void CHtest();
    static void YJtest();
    static void WLtest();
    static void PCtest();


};





#endif
