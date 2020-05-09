#include "randompartition.h"
#include "nngkmeans.h"
#include "ioagent.h"
#include "vstring.h"
#include "timer.h"
#include "nn.h"

#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <map>

///write by pclin 2018-3-3
#include<time.h>

using namespace std;

static unsigned mx = 50;
//static unsigned rvn = 20;
///static unsigned Check = 10;
static unsigned topRange = 0;
static unsigned p = 50;

#define SIFT

const unsigned NNGKMeans::RangeFirst = mx+p;
const unsigned NNGKMeans::RangeSecond = topRange+p;

const unsigned int NNGKMeans::NIter   = 30;///
/**/
const unsigned int NNGKMeans::TreeNum = 10;
///static unsigned ENum = 0;
const float NNGKMeans::Err0           = 0;
const unsigned long NNGKMeans::Treesize   = 50;
const unsigned long NNGKMeans::TopK       = 50;
const unsigned long NNGKMeans::NNTop      = 25;

const unsigned NNGKMeans::NBIter = 1;
/**/

//static unsigned TreeNum = 10, TopK = 25, NBIter = 1, Treesize = 50;
static unsigned t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, titer = 0, titer0 = 0;
static double rc;

bool static inline fuLs(const pair<float, unsigned> &a, const pair<float, unsigned> &b)
{
    if(a.first == b.first)
        return a.second < b.second;
    return a.first < b.first;
}

NNGKMeans::NNGKMeans()
{
    _INIT_ = false;
    this->data      = NULL;
    this->Ds        = NULL;
    this->Ns        = NULL;
    this->Es        = NULL;
    this->arrayD    = NULL;
    this->infoMap   = NULL;
    this->visitList = NULL;
    this->radius    = NULL;
    this->topDist   = NULL;
    this->topList   = NULL;
    this->prUpdated = NULL;
    this->crUpdated = NULL;
    kmMtd   = _xtkmn_;

    this->tmLogs  = NULL;
    this->dstLogs = NULL;

    strcpy(mthStr, "_gkmeans_");
    cout<<"Method ........................... GK-means\n";
}

bool NNGKMeans::setLogOn(const unsigned logSize)
{
   assert(logSize > 0);

   nLogs = logSize;
   if(this->tmLogs != NULL)
   {
       delete [] this->tmLogs;
       this->tmLogs = new double[logSize];
   }
   if(this->dstLogs != NULL)
   {
       delete [] dstLogs;
       dstLogs = new double[logSize];
   }
   return 1;
}

bool NNGKMeans::init(const char *srcfn)
{
    cout<<"Loading matrix ................... ";
    assert(srcfn);
    strcpy(srcmatfn, srcfn);
    unsigned ct = 0, nd = 0, i = 0;

    if(VString::endWith(srcfn, ".txt"))
    {
        ///ndim is the dim of the dataset
        this->data     = IOAgent::loadMatrix(srcfn, ct, nd);
        this->count    = ct;
        this->ndim     = nd;
        this->dataType = 0;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }
    else if(VString::endWith(srcfn, ".fvecs"))
    {
        this->data  = IOAgent::load_fvecs(srcfn, ct, nd);
        this->count = ct;
        this->ndim  = nd;
        this->dataType = 0;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }
    else if(VString::endWith(srcfn, ".bvecs"))
    {
        this->data  = IOAgent::load_bvecs(srcfn, ct, nd);
        this->count = ct;
        this->ndim  = nd;
        this->dataType = 0;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }
    else if(VString::endWith(srcfn, ".mat"))
    {
        this->data = IOAgent::loadDat(srcfn, ct, nd);
        this->count = ct;
        this->ndim  = nd;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }
    else if(VString::endWith(srcfn, ".tvecs"))
    {
        vector<unsigned int> ids;
        this->data = IOAgent::loadTruncMat(srcfn, ct, nd, idx2ids);
        this->count = ct;
        this->ndim  = nd;
    }else if(VString::endWith(srcfn, ".imat"))
    {
        this->data = IOAgent::loadIMat(srcfn, ct, nd);
        this->count = ct;
        this->ndim  = nd;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }
    else
    {
        this->data = IOAgent::loadItms(srcfn, "fsvtab", this->num, ct, nd);
        this->count = ct;
        this->ndim = nd;
        for(i = 0; i < ct; i++)
        {
            idx2ids.push_back(i);
        }
    }

    if(this->data == NULL)
    {
        cout<<"Exceptions ....................... Loading matrix failed!\n";
        exit(0);
    }

    cout<<this->count<<"x"<<this->ndim<<endl;


    this->_INIT_  = true;
    this->_REFER_ = false;

    ///cout<<"clust num\t"<<this->clnumb<<endl;
    cout<<"cluster Num. ..................... "<<this->clnumb<<endl;
    return true;
}

bool NNGKMeans::init(float *mat, const int row, const int dim)
{
    assert(mat);
    if(this->data != NULL)
    {
        cout<<"warning : reuse data before\n";
        return -1;
    }
    this->data   = mat;
    this->count  = row;
    this->ndim   = dim;

    cout<<row<<"x"<<dim<<endl;

    this->_INIT_ = true;

    ///cout<<"clust num\t"<<this->clnumb<<endl;
    return true;
}

bool NNGKMeans::config(const char *_seed_, const char *lg_first, const char *crtrn, int verbose)
{
    if(verbose)
        cout<<"Distance function ................ l2\n";

    if(verbose)
        cout<<"Seeds ............................ ";
    if(!strcmp(_seed_, "rnd"))
    {
        if(verbose)
            cout<<"rand\n";
        seed = _rnd_;
    }
    else if(!strcmp(_seed_, "kpp"))
    {

        if(verbose)
            cout<<"kpp\n";
        seed = _kpp_;
    }
    else
    {
        if(verbose)
            cout<<"non\n";
        seed = _non_;
    }

    if(verbose)
        cout<<"Optimization function ............ ";
    if(!strcmp(crtrn, "i1"))
    {
        myoptz     = _I1_;
        if(verbose)
            cout<<"I1\n";
    }
    else if(!strcmp(crtrn, "i2"))
    {
        myoptz  = _I2_;
        if(verbose)
            cout<<"I2\n";
    }
    else if(!strcmp(crtrn, "i3"))
    {
        myoptz  = _I3_;
        if(verbose)
            cout<<"I3\n";
    }
    else if(!strcmp(crtrn, "i4") )
    {
        myoptz  = _I4_;
        if(verbose)
            cout<<"e1\n";
    }
    else if(!strcmp(crtrn, "e1") )
    {
        myoptz  = _E1_;
        if(verbose)
            cout<<"e1\n";
    }
    else if(!strcmp(crtrn, "e2") )
    {
        myoptz  = _E2_;
        if(verbose)
            cout<<"e2\n";
    }
    else if(!strcmp(crtrn, "t1"))
    {
        myoptz = _T1_;
        if(verbose)
            cout<<"t1\n";
    }
    else if(!strcmp(crtrn, "i4"))
    {
        myoptz = _I4_;
        if(verbose)
            cout<<"i4\n";
    }
    else
    {
        cout<<"Unkown optimize option '"<<crtrn<<"'!\n";
        exit(0);
    }

    return true;
}

int NNGKMeans::updateNNGraph()
{
    unsigned sz = 0;
    vector<unsigned> *clusters = NULL;
    clusters = new vector<unsigned>[this->clnumb];
    unsigned long i = 0, j = 0, k = 0, id, r, col, r1 = 0, r2 = 0;
    float *cdst = NULL;
    float dst = 0;

    this->label2cluster(clusters);

    unsigned mxsize = 0;
    for(i = 0; i < this->clnumb; i++)
    {
        if(clusters[i].size() > mxsize)
            mxsize = clusters[i].size();
    }

    cdst = new float[mxsize*mxsize];

    for(i = 0; i < this->clnumb; i++)
    {
        ///Firstly we cal the distance between clusters and sample.
        sz     = clusters[i].size();
        for(j = 0; j < sz; j++)
        {
            id = clusters[i][j];
            for(k = j + 1; k < sz; k++)
            {
                r = clusters[i][k];
                dst = 0;
                r1  = r*ndim;
                r2  = id*ndim;
                for(col = 0; col < ndim; col++)
                {
                    dst += (data[r2+col] - data[r1+col])*(data[r2+col] - data[r1+col]);
                }
                cdst[j*sz+k] = dst;
            }
        }
        ///Secondly. we try to insert into the Nearest Neighbor list.
        for(j = 0; j < sz; j++)
        {
            id = clusters[i][j];
            for(k = j + 1; k < sz; k++)
            {
                r = clusters[i][k];
                dst = cdst[j*sz+k];
                if(dst <= radius[id])
                    InsertIntoKnn(topList+id*TopK, topDist+id*TopK, TopK, dst, r, radius[id]);
                if(dst <= radius[r])
                    InsertIntoKnn(topList+r*TopK, topDist+r*TopK, TopK, dst, id, radius[r]);
            }
        }
        clusters[i].clear();
    }
    this->adjustNN();
    delete [] cdst;
    cdst = NULL;
    delete [] clusters;
    clusters = NULL;
    ///cout<<"update top finished\n";
    return 0;
}

int NNGKMeans::adjustNN()
{
    unsigned long i, j;
    vector<pair<float, unsigned> > topmap;
    topmap.resize(TopK);
    for(i = 0; i < this->count; i++)
    {
        for(j = 0; j < this->TopK; j++)
        {
            topmap[j].second = topList[i*TopK+j];
            topmap[j].first = topDist[i*TopK+j];
        }
        sort(topmap.begin(), topmap.end(), fuLs);
        for(j = 0; j < this->TopK; j++)
        {
            topList[i*TopK+j] = topmap[j].second;
            topDist[i*TopK+j] = topmap[j].first;
        }
    }
    return  0;
}

int NNGKMeans::initNNGraph()
{
    unsigned long i, j, k, id, rid, r1 = 0, r2 = 0;
    float dst = 0, mxdst = 0;
    unsigned *rl = new unsigned[this->count];
    //random numbers .
    for(i = 0; i < this->count; i++)
    {
        rl[i] = i;
    }
    //
    random_shuffle(rl, rl+this->count);
    ///for each node sample,initial its k-nearest neighbors
    for(i = 0; i < this->count; i++)
    {
        rid = rand()%this->count;
        mxdst = 0;
        for(j = 0; j < TopK; j++)
        {
            //random select a id from random queue
            id = rl[(j+rid)%this->count];

            if(id == i)
            {
                id = rl[(rid+TopK+1)%this->count];
            }
            //toplist is a two dimension metrix to store knngraph
            topList[i*TopK+j] = id;
            dst = 0;
            r1  = i*this->ndim;
            r2  = id*this->ndim;
            for(k = 0; k < this->ndim; k++)
            {
                dst += (data[r1 + k]-data[r2+k])*(data[r1+k]-data[r2+k]);
            }
            topDist[i*TopK+j] = dst;
            if(dst > mxdst)
                mxdst = dst;
        }
        radius[i] = mxdst;
    }

    delete [] rl;
    rl = NULL;
    return 0;
}

int NNGKMeans::buildNNGraphBy2Mn(int verbose)
{
    unsigned clustNum = 0, i = 0;
    RandomPartition *rpTree = NULL;
    Timer *itm = NULL;
    /// init
    clustNum = this->clnumb;
    this->clnumb = this->count/Treesize;
    this->clnumb = pow(2, floor(log(this->clnumb)/log(2)));

    if(this->Ds != NULL)
    {
        free(this->Ds);
        this->Ds = NULL;
    }
    if(this->Ns != NULL)
    {
        free(this->Ns);
        this->Ns = NULL;
    }
    if(this->arrayD != NULL)
    {
        free(this->arrayD);
        this->arrayD = NULL;
    }
    if(this->Es != NULL)
    {
        free(this->Es);
        this->Es = NULL;
    }

    for(i = 0; i < TreeNum; i++)
    {
        rpTree = new RandomPartition;
        rpTree->buildcluster(this->data, this->count, this->ndim, "tmpfile.txt",  "rnd", "large", "i2", this->clnumb, 0);
        rpTree->getLabel(this->labels);
        delete rpTree;
        rpTree = NULL;

        if(verbose)
        {
           itm = new Timer;
           itm->start();
        }
        if(i > 0)
        {
            this->Es     = (double*)calloc(this->clnumb, sizeof(double));//energy of each cluster .
            this->arrayD = (double*)calloc(this->clnumb, this->ndim*sizeof(double));//
            this->Ds     = (double*)calloc(this->clnumb, this->ndim*sizeof(double));//center of cluster
            this->Ns     = (int*)calloc(this->clnumb, sizeof(int)); //numbers of each cluster

            ///-----------------------------------------------//
            ///optimize the cluster using the I2 objective function.
            this->optzI2(0, NNTop, NBIter);
            ///--------------------------------------------------///
            ///------------------free operations-----------------///
            free(this->Ds);
            this->Ds = NULL;
            free(this->Ns);
            this->Ns = NULL;
            free(this->arrayD);
            this->arrayD = NULL;
            free(this->Es);
            this->Es = NULL;
            ///-------------------------------------------------///
        }

        if(verbose)
        {
           itm->end(true);
        }
        updateNNGraph();

        ///cout<<"recall:"<<recall()<<endl;
    }

    this->clnumb = clustNum;
    return true;
}

int NNGKMeans::recall()
{
    int i = 0, j = 0, id = 0;
    /**/
    unsigned tmpid[200];
    unsigned *rtop = new unsigned[200*this->count];
    ifstream is;
    if(this->count == 1000000 && this->ndim == 128)
        is.open("sift_base200nn.txt");
    else if(this->count == 100000 && this->ndim == 128)
        is.open("nn.txt");
    else if(this->count >= 5000 && this->count <= 6000 && this->ndim == 2048)
        is.open("ox5k200nn.txt");
    else if(this->count == 50000 && this->ndim == 2048)
        is.open("ox50k200nn.txt");
    else if(this->count >= 100000 && this->count <= 100000+1000 && this->ndim == 2048)
        is.open("ox100k200nn.txt");
    else if(this->count == 500000 && this->ndim == 960)
        is.open("gist_learn200nn.txt");
    else if(this->count == 1000000 && this->ndim == 960)
        is.open("gist_200nn.txt");
    else
    {
        cout<<"erro no ground truth for count "<<this->count<<endl;
        return -1;
    }
    if(!is.is_open())
    {
        cout<<"cant not open groundturth \n";
        return -1;
    }

    for(i = 0; i < this->count; i++)
    {
        for(j = 0; j < 200; j++)
        {
            is>>id;
            tmpid[j] = id;
        }
        for(j = 0; j < 200; j++)
            rtop[i*200+j] = tmpid[j];
    }
    is.close();
    unsigned r[4] = {0};
    unordered_set<unsigned> tmpset;
    unsigned b[4] = {1,3,5,10};
    unsigned a = 0;
    for(i = 0; i < this->count; i++)
    {
        for(id = 0; id < 4; id++)
        {
            a = b[id];
            tmpset.clear();
            for(j = 0; j < TopK; j++)
            {
                if(j >= a)
                    break;
                tmpset.insert(topList[i*TopK+j]);
            }
            for(j = 0; j < 1; j++)
            {
                tmpset.insert(rtop[i*200+j]);
            }
            if(tmpset.size() < 1+a)
            {
                r[id]++;
            }
            if(a == 0)
                cout<<a<<endl;
        }
    }
    unsigned cr[5] = {0};
    unsigned c[5] = {5,10,25,50,100};
    for(i = 0; i < this->count; i++)
    {
        for(id = 0; id < 5; id++)
        {
            a = c[id];
            tmpset.clear();
            for(j = 0; j < TopK; j++)
            {
                if(j >=a)
                    break;
                tmpset.insert(topList[i*TopK+j]);
            }
            for(j = 0; j < a; j++)
            {
                tmpset.insert(rtop[i*200+j]);
            }
            cr[id] += 2*a - tmpset.size();
        }
    }
    double ctop5  = (cr[0]+0.0)/this->count/5;
    double ctop10 = (cr[1]+0.0)/this->count/10;//10;
    double ctop15 = (cr[2]+0.0)/this->count/25;//50;
    double ctop20 = (cr[3]+0.0)/this->count/50;//100;
    double ctop25 = (cr[4]+0.0)/this->count/100;//100;

    double top1   = (r[0]+0.0)/this->count;
    double top10  = (r[1]+0.0)/this->count;//10;
    double top50  = (r[2]+0.0)/this->count;//50;
    double top100 = (r[3]+0.0)/this->count;//100;
    cout<<"recall 1 "<<top1<<" 3 "<<top10<<" 5 "<<top50<<" 10 "<<top100<<endl;
    cout<<"cover 5 "<<ctop5<<" 10 "<<ctop10<<" 25 "<<ctop15<<" 50 "<<ctop20<<" 100 "<<ctop25<<endl;

    /**/

    delete [] rtop;
    rc = top1;
    return 0;
}

int NNGKMeans::buildNNGraph4clust()
{
    double Ttime = 0;
    clock_t start = 0, finish = 0;
    start = clock();
    this->topList = (unsigned*)calloc(this->count, TopK*sizeof(unsigned));
    this->topDist = (float*)calloc(this->count, TopK*sizeof(float));
    this->radius  = (float*)calloc(this->count, sizeof(float));

    cout<<"Build k-NN graph by GK-means ..... nTree = "<<TreeNum<<endl;
    initNNGraph();
    buildNNGraphBy2Mn(0);
    finish = clock();
    Ttime = (double)(finish-start)/CLOCKS_PER_SEC;
    this->tmptime = Ttime;

    free(topDist);
    free(radius);
    this->topDist = NULL;
    this->radius  = NULL;

    return 0;
}

int NNGKMeans::clust(const unsigned int clust_num, const char *dstfn, const int verbose)
{
    unsigned long clustNum = 0, i = 0;
    RandomPartition *xbk = NULL;
    titer0 = clock();

    buildNNGraph4clust();

    this->Es     = (double*)calloc(this->clnumb, sizeof(double));
    this->arrayD = (double*)calloc(this->clnumb, this->ndim*sizeof(double));
    this->Ds     = (double*)calloc(this->clnumb, this->ndim*sizeof(double));
    this->Ns     = (int*)calloc(this->clnumb, sizeof(int));

    for(i = 0; i < this->clnumb; i++)
    {
        memset(Ds+i*this->ndim, 0, sizeof(double)*this->ndim);
        memset(Ns+i, 0, sizeof(int));
    }

    cout<<"Initialize cluster by 2-Means .... ";
    xbk = new RandomPartition;
    xbk->buildcluster(this->data, this->count, this->ndim, "tmpfile.txt",  "rnd", "large", "i2", this->clnumb, 0);
    xbk->getLabel(this->labels);
    delete xbk;
    cout<<"done\n";

    ///cout<<"opt begin!"<<endl;
    this->optzI2(1, this->TopK, NIter);
    this->calAVGDist(this->arrayD, this->Ns, clust_num, this->infoMap);

    ///cout<<"end!"<<endl;
    this->save_clust(dstfn);

    free(arrayD);
    free(Es);
    free(Ds);
    free(Ns);
    arrayD = NULL;
    Es = NULL;
    Ds = NULL;
    Ns = NULL;
    return clustNum;
}

int NNGKMeans::NNI2(bool verbose, const unsigned iter0)
{
    if(iter0 < 1)
        return 0;

    double delta, delta0, optEg, tmpEg = 0, tmpEg1 = 0, tmpEg2 = 0, len, allEg;
    unsigned long i = 0, j = 0, k = 0, iter = 0, id = 0, row = 0;
    int label, nlabel = 0;
    float *tmpData = NULL;
    bool UPDATE = false;
    unordered_set<unsigned> tmptop;
    allEg = 0;
    for(i = 0; i < this->count; i++)
    {
        ///allEg can be cal before clustering.
        allEg += lens[i];
        label  = this->labels[i];
        Ns[label]++;
        row   = i*this->ndim;
        for(j = 0; j < this->ndim; j++)
        {
            Ds[label*this->ndim+j] += data[row+j];
        }
    }
    for(i = 0; i < this->clnumb; i++)
    {
        Es[i] = 0;
        row   = i*this->ndim;
        for(j = 0; j < this->ndim; j++)
        {
            ///sum of (Dr*Dr)...
            Es[i] += this->Ds[row+j]*this->Ds[row+j];
        }
    }
    //getI2 return  sum of Dr'*Dr/nr r belong to [1,k]
    optEg = getI2(this->Ds, this->clnumb, this->ndim, Ns);
    iter = 0;

    do
    {
        UPDATE = false;
        for(i = 0; i < this->count; i++)
        {
            ///only compare with active points
            id = i;
            label = labels[id];
            nlabel = label;
            tmptop.clear();

            for(auto it = visitList[i].begin(); it != visitList[i].end(); it++)
            {
                if(label != labels[*it])
                    tmptop.insert(labels[*it]);
            }

            if(tmptop.empty())
                continue;

            if(Ns[label] <= 1)
            {
                continue;
            }
            len = this->lens[id];
            tmpData = this->data + id*this->ndim;
            ///only compare with neiborhood cluster;
            delta0 = 0;
            tmpEg1 = 0;
            for(j = 0; j < ndim; j++)
            {
                tmpEg1 += Ds[label*this->ndim+j]*tmpData[j];
            }
            tmpEg1 = Es[label] - 2*tmpEg1 + len;
            for(auto it = tmptop.begin(); it != tmptop.end(); it++)
            {
                k = *it;
                if(k == label)
                    continue;
                tmpEg2 = 0;
                for(j = 0; j < ndim; j++)
                {
                    tmpEg2 += Ds[k*this->ndim+j]*tmpData[j];
                }
                tmpEg2 = Es[k] + 2*tmpEg2 + len;
                delta  = tmpEg2/(Ns[k]+1) - Es[k]/Ns[k] + tmpEg1/(Ns[label]-1) - Es[label]/Ns[label];
                if(delta > delta0)
                {
                    delta0 = delta;
                    nlabel = k;
                    tmpEg = tmpEg2;
                }
            }
            if(delta0 > 0)
            {
                UPDATE = true;
                ///update points belong to new cluster
                Ns[label]--;
                Ns[nlabel]++;
                for(j = 0; j < this->ndim; j++)
                {
                    Ds[label*this->ndim+j] -= tmpData[j];
                    Ds[nlabel*this->ndim+j] += tmpData[j];
                }
                Es[label]  = tmpEg1;
                Es[nlabel] = tmpEg;
                labels[id] = nlabel;
            }
        }

        iter++;
        optEg = getI2(this->Ds, this->clnumb, this->ndim, Ns);
    }while(iter < iter0 && UPDATE);

    for(i = 0; i < this->clnumb; i++)
    {
        memcpy(this->arrayD+i*this->ndim, Ds + i*this->ndim, sizeof(double)*this->ndim);
    }

    return clnumb;
}

int NNGKMeans::optzI2(bool verbose, const unsigned NNTop, const unsigned iter0)
{
    if(iter0 < 1)
        return 0;

    double delta, delta0, optEg, tmpEg = 0, tmpEg1 = 0, tmpEg2 = 0, len, allEg;
    unsigned long i = 0, j = 0, k = 0, iter = 0, id = 0, row = 0;
    unsigned int status = 0, nIgrs = 0, nDups = 0;
    int label = 0, nlabel = 0;
    float *tmpData = NULL;
    bool UPDATE = false;
    unordered_set<unsigned> tmptop;
    allEg = 0;

    this->prUpdated = new unsigned char[this->clnumb];
    this->crUpdated = new unsigned char[this->clnumb];
    memset(this->prUpdated, 0, this->clnumb);
    memset(this->crUpdated, 0, this->clnumb);

    ///initial the cluster first.
    for(i = 0; i < this->count; i++)
    {
        allEg += lens[i];
        label = this->labels[i];
        Ns[label]++;
        row   = i*this->ndim;
        for(j = 0; j < this->ndim; j++)
        {
            Ds[label*this->ndim+j] += data[row+j];
        }
    }
    for(i = 0; i < this->clnumb; i++)
    {
        Es[i] = 0;
        row   = i*this->ndim;
        for(j = 0; j < this->ndim; j++)
        {
            Es[i] += this->Ds[row+j]*this->Ds[row+j];
        }
    }

    optEg = getI2(this->Ds, this->clnumb, this->ndim, Ns);
    iter = 0;
    if(verbose)
       cout<<"iter "<<iter<<"\toptEg "<<optEg<<"\tavgdistortion "<<(allEg -optEg)/this->count<<endl;

    memset(this->crUpdated, 1, this->clnumb);

    do
    {
        UPDATE = false;
        memcpy(this->prUpdated, this->crUpdated, this->clnumb);
        memset(this->crUpdated, 0, this->clnumb);
        for(i = 0; i < this->count; i++)
        {
            ///only compare with active points
            id     = i;
            label  = labels[id];
            nlabel = label;
            tmptop.clear();

            for(j = 0; j < NNTop; j++)
            {
                if(label != labels[topList[id*TopK+j]])
                    tmptop.insert(labels[topList[id*TopK+j]]);
            }

            if(tmptop.empty())
            {
                nDups++;
                continue;
            }

            if(Ns[label] <= 1)
            {
                continue;
            }
            len = this->lens[id];
            tmpData = this->data + id*this->ndim;
            ///only compare with neiborhood cluster;
            delta0 = 0;
            tmpEg1 = 0;
            for(j = 0; j < ndim; j++)
            {
                tmpEg1 += Ds[label*this->ndim+j]*tmpData[j];
            }
            tmpEg1 = Es[label] - 2*tmpEg1 + len;
            status = 0;
            for(auto it = tmptop.begin(); it != tmptop.end(); it++)
            {
                status += this->prUpdated[*it];
            }

            if(status < 1)
            {
                nIgrs++;
                continue;
            }/**/

            for(auto it = tmptop.begin(); it != tmptop.end(); it++)
            {
                k = *it;
                tmpEg2 = 0;
                for(j = 0; j < ndim; j++)
                {
                    tmpEg2 += Ds[k*this->ndim+j]*tmpData[j];
                }
                tmpEg2 = Es[k] + 2*tmpEg2 + len;
                delta  = tmpEg2/(Ns[k]+1) - Es[k]/Ns[k] + tmpEg1/(Ns[label]-1) - Es[label]/Ns[label];
                if(delta > delta0)
                {
                    delta0 = delta;
                    nlabel = k;
                    tmpEg = tmpEg2;
                }
            }

            if(delta0 > 0)
            {
                UPDATE = true;
                ///update points belong to new cluster
                Ns[label]--;
                Ns[nlabel]++;
                for(j = 0; j < this->ndim; j++)
                {
                    Ds[label*this->ndim+j]  -= tmpData[j];
                    Ds[nlabel*this->ndim+j] += tmpData[j];
                }
                Es[label]  = tmpEg1;
                Es[nlabel] = tmpEg;
                labels[id] = nlabel;
                this->crUpdated[label]  = 1;
                this->crUpdated[nlabel] = 1;
            }
        }

        iter++;
        optEg = getI2(this->Ds, this->clnumb, this->ndim, Ns);

        /**/
        if(verbose)
           cout<<"iter "<<iter<<"\toptEg "<<optEg<<"\tavgdistortion "<<(allEg -optEg)/this->count<<"\tIgnores: "<<nIgrs<<"\tdups: "<<nDups<<endl;
        /**/
    }while(iter < iter0 && UPDATE);

    for(i = 0; i < this->clnumb; i++)
    {
        memcpy(this->arrayD+i*this->ndim, Ds + i*this->ndim, sizeof(double)*this->ndim);
    }

    delete [] prUpdated;
    prUpdated = NULL;
    delete [] crUpdated;
    crUpdated = NULL;

    return clnumb;
}

int NNGKMeans::buildKNNGraph(const char* srcfn, const char* dstfn, const int dpg)
{
    this->init(srcfn);
    this->initMemry(this->count, 0);
    this->buildNNGraph4clust();
    //this->count = 100000000;
    //topList = this->loadKNNGraph(srcfn);
    this->saveKNNGraph(dstfn, this->TopK);
    assert(dpg < TopK);
    if(dpg > 0)
    {
        this->diversify_by_cut(topList, this->count, TopK, dpg);
        this->augRvsKNN(topList, this->count, TopK, dpg);
    }

    //this->saveKNNGraphDPG(dstfn, this->TopK);

    return 0;
}

void NNGKMeans::augRvsKNN(unsigned int *kNNGraph, unsigned long N, unsigned long D,
                                 unsigned int L)
{
    unsigned int i = 0, j = 0, knn, step = N/1000;
    unsigned long loc = 0;
    unsigned *lastMaxSZ  = NULL;
    unsigned *tmpList    = NULL;
    unsigned addSZ = 25;
    bool DUP = false;
    rkNNNum   = new unsigned [N];
    memset(rkNNNum, 0, N*sizeof(unsigned));
    lastMaxSZ = new unsigned [N];
    rkNNGraph = (unsigned **) malloc(N*sizeof(unsigned *));
    for(i = 0; i < N; i++)
    {
        lastMaxSZ[i] = 25;
        rkNNGraph[i] = (unsigned *) malloc(25 * sizeof(unsigned));
    }

    for(i = 0; i < N; i++)
    {
        loc = i*D;
        if((i) % step == 0)
        {
            cout << "\r\r\r\r\t" << loc;
        }

        for(j = 0; j < L; j++)
        {
            knn = kNNGraph[loc+j];
            rkNNGraph[knn][rkNNNum[knn]] = i;
            rkNNNum[knn]++;
            if(rkNNNum[knn] >= lastMaxSZ[knn])
            {
                tmpList = (unsigned *) realloc (rkNNGraph[knn], (addSZ + lastMaxSZ[knn]) * sizeof(unsigned));
                rkNNGraph[knn] = tmpList;
                lastMaxSZ[knn] = addSZ + lastMaxSZ[knn];
            }
        }
    }

    for(i = 0; i < N; i++)
    {
        loc = i*D;
        for(j = 0; j < L; j++)
        {
           knn = kNNGraph[loc+j];
           DUP = false;
           for(int k = 0; k < rkNNNum[i]; k++)
           {
               if(rkNNGraph[i][k] == knn)
               {
                   DUP = true;
                   break;
               }
           }
           if(DUP)
                continue;

           rkNNGraph[i][rkNNNum[i]] = knn;
           rkNNNum[i]++;
           if(rkNNNum[i] >= lastMaxSZ[i])
            {
                tmpList = (unsigned *) realloc (rkNNGraph[i], (addSZ + lastMaxSZ[i]) * sizeof(unsigned));
                rkNNGraph[i] = tmpList;
                lastMaxSZ[i] = lastMaxSZ[i] + addSZ;
            }
        }
    }

    saveKNNGraphDPG("nusw_gkmeans_l2.txt", N, D, rkNNGraph, rkNNNum);

    return ;
}

int NNGKMeans::buildCluster(unsigned* NN, float* data, unsigned row, unsigned ndim, unsigned clustnumb)
{
    unsigned i = 0;
    this->init(data, row, ndim);
    this->clnumb = clustnumb;
    this->initMemry(row, clustnumb);
    this->copyFromTopList(NN, 25);
    //--------------------------------------------------------------------//
    this->Es     = (double*)calloc(this->clnumb, sizeof(double));
    this->arrayD = (double*)calloc(this->clnumb, this->ndim*sizeof(double));
    this->Ds     = (double*)calloc(this->clnumb, this->ndim*sizeof(double));
    this->Ns     = (int*)calloc(this->clnumb, sizeof(int));
    //--------------------------------------------------------------------//
    for(i = 0; i < this->clnumb; i++)
    {
        memset(Ds+i*this->ndim, 0, sizeof(double)*this->ndim);
        memset(Ns+i, 0, sizeof(int));
    }
    cout<<"Initialize cluster begin .......\n";
    RandomPartition *xbk = new RandomPartition;
    xbk->buildcluster(this->data, this->count, this->ndim, "tmpfile.txt",  "rnd", "large", "i2", this->clnumb, 0);
    xbk->getLabel(this->labels);
    delete xbk;
    ///cout<<"initialize cluster finished all\n";
    Timer *mytm2 = new Timer();
    mytm2->start();
    this->optzI2(0, this->TopK, NIter);
    mytm2->end(true);
    this->calAVGDist(this->arrayD, this->Ns, clnumb, this->infoMap);

    free(Es);
    free(arrayD);
    free(Ds);
    free(Ns);
    Es = NULL;
    arrayD = NULL;
    Ds = NULL;
    Ns = NULL;

    return 0;
}

int NNGKMeans::buildCluster(const char *srcfn, const char *nnfn, const char *dstfn, unsigned clustnumb)
{
    clock_t start_i,finish_i;
    unsigned row = 0, dim = 0, i = 0;
    this->init(srcfn);
    start_i = clock();

    this->clnumb = clustnumb;
    this->initMemry(this->count, clustnumb);

    this->Es     = (double*)calloc(this->clnumb, sizeof(double));// energy of each cluster .double
    this->arrayD = (double*)calloc(this->clnumb, this->ndim*sizeof(double)); //center of each cluster.and its dimension
    this->Ds     = (double*)calloc(this->clnumb, this->ndim*sizeof(double)); //center of each cluster and its dimension
    this->Ns     = (int*)calloc(this->clnumb, sizeof(int));     //numbers of samples in each cluster.

    finish_i = clock();
    double tpTime = (finish_i-start_i)/CLOCKS_PER_SEC;
    this->tmptime += tpTime;

    this->topList = IOAgent::loadNN(nnfn, row, dim);
    start_i = clock();

    if(row != this->count || dim != this->TopK)
    {
        cout<<"erro , not match : count = "<<this->count<<" , row = "<<row<<" dim = "<<dim<<" topk = "<<this->TopK<<endl;
        return -1;
    }

    for(i = 0; i < this->clnumb; i++)
    {
        memset(Ds+i*this->ndim, 0, sizeof(double)*this->ndim);
        memset(Ns+i, 0, sizeof(int));
    }
    RandomPartition *xbk = new RandomPartition;
    xbk->buildcluster(this->data, this->count, this->ndim, "tmpfile.txt",  "rnd", "large", "i2", this->clnumb, 0);
    xbk->getLabel(this->labels);
    delete xbk;
    Timer *mytm2 = new Timer();
    mytm2->start();

    this->optzI2(0, this->TopK, this->NIter);
    mytm2->end(true);
    this->calAVGDist(this->arrayD, this->Ns, clnumb, this->infoMap);
    ///save clust
    ///  this->save_clust(dstfn);
    finish_i = clock();
    tpTime   = (finish_i-start_i)/CLOCKS_PER_SEC;
    this->tmptime += tpTime;
    free(arrayD);
    free(Es);
    free(Ds);
    free(Ns);
    arrayD = NULL;
    Es = NULL;
    Ds = NULL;
    Ns = NULL;
    return 0;
}

int NNGKMeans::copy2TopList(unsigned *top)
{
    for(unsigned i = 0; i < this->count; i++)
    {
        memcpy(top+i*TopK, this->topList+i*TopK, sizeof(unsigned)*TopK);
    }
    return 0;
}

int NNGKMeans::copyFromTopList(unsigned *top, unsigned topNum)
{
    this->topList = (unsigned*)calloc(this->count, TopK*sizeof(unsigned));
    for(unsigned i = 0; i < this->count; i++)
    {
        memcpy(this->topList+i*TopK, top+i*TopK, sizeof(unsigned)*TopK);
    }
    return 0;
}

void NNGKMeans::saveCenters(const char *dstfn, bool append)
{
    unsigned long clabel = 0, i, j, loc, rCNum = 0;
    bool isNULL = false;
    ofstream *outStrm  = NULL;

    if(this->arrayD == NULL)
    {
        isNULL = true;
        this->arrayD = (double*)calloc(this->clnumb, this->ndim*sizeof(double));
    }

    for(i = 0; i < count; i++)
    {
        clabel = labels[i];
        for(j = 0; j < ndim; j++)
        {
            arrayD[clabel*ndim+j] += data[i*ndim+j];
        }
    }

    for(clabel = 0; clabel < this->clnumb; clabel++)
    {
        if(this->infoMap[clabel].n > 0)
        {
            rCNum++;
        }
    }

    if(!this->_INIT_||rCNum == 0)
    {
        return ;
    }

    if(append)
    {
        outStrm = new ofstream(dstfn, ios::app);
    }
    else
    {
        outStrm = new ofstream(dstfn, ios::out);
        (*outStrm)<<rCNum<<" "<<this->ndim<<endl;;
    }
    for(clabel = 0; clabel < this->clnumb; clabel++)
    {
        loc  = clabel*this->ndim;

        if(this->infoMap[clabel].n <= 0)
            continue;
        for(j = 0; j < this->ndim; j++)
        {
            (*outStrm)<<this->arrayD[loc+j]/this->infoMap[clabel].n<<" ";
        }
        (*outStrm)<<endl;
    }

    outStrm->close();

    if(isNULL)
    {
        free(arrayD);
        this->arrayD = NULL;
    }

    cout<<"done\n";
    return ;
}

float NNGKMeans::l2(const unsigned int i, const unsigned int j)
{
    float dst = 0, diff = 0;
    unsigned int k = 0, loc1 = i*ndim, loc2 = j*ndim;
    for(k = 0; k < this->ndim; k++)
    {
        diff = this->data[loc1+k] - this->data[loc2+k];
        dst += diff*diff;
    }
    return dst;
}

void NNGKMeans::diversify_by_cut(unsigned int *kNNGraph, const unsigned int N, const unsigned int K, const int edge_num)
{
    float *dsts = new float[2*edge_num], dist = 0, cut = 0;
    unsigned int *tmp = new unsigned int[2*edge_num];
    int* b_hit = new int[2*edge_num];
    int* hit = new int[2*edge_num];
    int cnt = 0, i = 0, j = 0;
    int len = K;

    unsigned step = N / 100, k = 0, n_b, n_k;
    unsigned long row = 0;
    cerr << endl << "Progress : ";
    for (k = 0; k < N; k++)
    {
      row = k*K;
      if (k % step == 0 )
         cerr << "*";

      if(len > 2*edge_num)
      {
        len = 2*edge_num;
      }

      /// materialize the ditance here
      for (i = 0; i < len; i++)
      {
        dsts[i] = l2(k, kNNGraph[row+i]);
        hit[i]  = 0;
      }

      for (i = 0; i < len-1; i++ )
      {
         for (j = i+1; j < len; j++)
         {

            if (i == j) continue;

            n_b = kNNGraph[row+i];
            n_k = kNNGraph[row+j];

            dist = l2(n_b, n_k);
            if (dist < dsts[j])
            {
                hit[j]++;
            }
        }
      }

      /// sort by the hits and find the cuts
      for(i = 0; i < len; i++ )
        b_hit[i] = hit[i];

      sort(b_hit, b_hit + len );
      cut = b_hit[edge_num];

      for (i = 0; i < len; i++)
        tmp[i] = kNNGraph[row+i];


      cnt = 0;
      for(i = 0; i < len; i++)
      {
        if(hit[i] <= cut )
        {
            kNNGraph[row+cnt++] = tmp[i];
        }
      }
    }

    delete [] hit;
    delete [] b_hit;
    delete [] tmp;
    delete [] dsts;
    dsts = NULL;
    hit = b_hit = NULL;
    tmp = NULL;

    cerr << endl;
}

void NNGKMeans::saveKNNGraph(const char *dstfn, unsigned int k0)
{
    ofstream *outStrm  = NULL;
    unsigned int i = 0, j = 0;
    outStrm = new ofstream(dstfn, ios::out);
    unsigned n = this->count, knn = 0;
    if(k0 > this->TopK)
        k0 = this->TopK;
    (*outStrm)<<n<<" "<<k0<<endl;
    for(i = 0; i < n; i++)
    {
        (*outStrm)<<idx2ids[i]<<" "<<k0<<" ";
        for(j = 0; j < k0; j++)
        {
            knn = topList[i*TopK+j];
            (*outStrm)<<" "<<idx2ids[knn];
        }
        (*outStrm)<<endl;
    }
    outStrm->close();
    cout<<"Done\n";
}

/// write by pclin 2018-2-3 for special case to store knn-graph
void NNGKMeans::saveKGraph(const char *dstfn,unsigned int tpk)
{
    ofstream *OutStrm = new ofstream(dstfn,ios::out);
    unsigned Trow = this->count;
    unsigned int irow = 0, idim = 0;
    if(tpk >this->TopK)
    {
        tpk = this->TopK;
    }
    (*OutStrm)<<Trow<<" ";
    (*OutStrm)<<tpk<<endl;
    for(irow = 0; irow <Trow ;irow++)
    {
        (*OutStrm)<<irow<<" "<<tpk<<" ";
        for(idim=0;idim <tpk;idim++)
        {
            unsigned knn = topList[irow*TopK+idim];
            (*OutStrm)<<idx2ids[knn]<<" ";
        }
        (*OutStrm)<<endl;
    }
    OutStrm->close();
    cout<<"savekgraph done!\n";
}

unsigned *NNGKMeans::loadKNNGraph(const char *srcfn)
{
    ifstream inStrm(srcfn, ios::in);
    unsigned N, tmp, k0;
    unsigned *topList;

    inStrm >> N;
    inStrm >> k0;
    cout << "kNN graph SZ = " << N << "x" << k0 << endl;
    topList = (unsigned*)calloc(N, k0*sizeof(unsigned));
    for(unsigned long i = 0; i < N; i++)
    {
        inStrm >> tmp;
        for(unsigned long j = 0; j < k0; j++)
        {
            inStrm >> topList[i*k0 + j];
        }
    }

    return topList;
}

void NNGKMeans::saveKNNGraphDPG(const char *dstfn, unsigned int k0)
{
    ofstream *outStrm  = NULL;
    unsigned int i = 0, j = 0, major = 2, minor = 0, k = 0;
    unsigned int knn = 0;
    char magic[9] = "KNNGRAPH";
    outStrm = new ofstream(dstfn, ios::out|ios::binary);
    unsigned n = this->count;
    outStrm->write((char*)&magic, 8);
    outStrm->write((char*)&major, sizeof(major));
    outStrm->write((char*)&minor, sizeof(minor));
    outStrm->write((char*)&n, sizeof(n));

    if(k0 > this->TopK)
        k0 = this->TopK;

    for(i = 0; i < n; i++)
    {
        k   = k0;
        knn = idx2ids[i];
        outStrm->write((char*)&knn, sizeof(unsigned int));
        outStrm->write((char*)&k,   sizeof(unsigned int));
        for(j = 0; j < k0; j++)
        {
            knn = topList[i*TopK+j];
            knn = idx2ids[knn];
            outStrm->write((char*)&knn, sizeof(unsigned int));
        }
    }

    outStrm->close();
    cout<<"done\n";
}

void NNGKMeans::saveKNNGraphDPG(const char *dstfn, const unsigned int N,
                                const unsigned int D, unsigned int **knnGraph, unsigned int *knnNumb)
{
    unsigned int i, j = 0, knn;
    ofstream *outStrm = new ofstream(dstfn, ios::out);
    (*outStrm)<<N<<endl;
    for(i = 0; i < N; i++)
    {
        (*outStrm)<<idx2ids[i]<<" "<<knnNumb[i];
        for(j = 0; j < knnNumb[i]; j++)
        {
            knn = knnGraph[i][j];
            (*outStrm)<<" "<<idx2ids[knn];
        }
        (*outStrm)<<endl;
    }
    outStrm->close();
    return ;
}

void NNGKMeans::saveKNNGraph(const char *dstfn, const unsigned int n, const unsigned int k0, vector<unsigned int> &ids)
{
    ofstream *outStrm  = NULL;
    unsigned int i = 0, j = 0, id;
    outStrm = new ofstream(dstfn, ios::out);
    (*outStrm)<<n<<"\t"<<k0<<endl;
    for(i = 0; i < n; i++)
    {
        (*outStrm)<<ids[i];
        for(j = 0; j < k0; j++)
        {
            id = topList[i*TopK+j];
            (*outStrm)<<" "<<ids[id];
        }
        (*outStrm)<<endl;
    }
    outStrm->close();
    cout<<"done\n";
}

int NNGKMeans::fetchCenters(float *centers)
{
    unsigned int clabel = 0, j = 0, loc = 0, idxi = 0, rCNum = 0;
    assert(centers);

    for(clabel = 0; clabel < this->clnumb; clabel++)
    {
        if(this->infoMap[clabel].n > 0)
        {
            rCNum++;
        }
    }

    if(!this->_INIT_||rCNum == 0)
    {
        memset(centers, 0, this->clnumb*this->ndim*sizeof(float));
        return 0;
    }

    for(clabel = 0; clabel < this->clnumb; clabel++)
    {
        loc   = clabel*this->ndim;

        if(this->infoMap[clabel].n <= 0)
            continue;

        for(j = 0; j < this->ndim; j++)
        {
            centers[idxi + j] = this->arrayD[loc+j]/this->infoMap[clabel].n;
        }
        idxi += this->ndim;
    }
    return rCNum;
}

NNGKMeans::~NNGKMeans()
{
    if(this->arrayD != NULL)
    {
        free(arrayD);
        this->arrayD = NULL;
    }
    if(this->Ds != NULL)
    {
        free(Ds);
        Ds = NULL;
    }
    if(this->Ns != NULL)
    {
        free(this->Ns);
        this->Ns = NULL;
    }
    if(this->Es != NULL)
    {
        free(this->Es);
        this->Es = NULL;
    }
    if(this->topDist != NULL)
    {
        free(topDist);
        topDist = NULL;
    }
    if(!this->outNN&&this->topList != NULL)
    {
        free(topList);
        topList = NULL;
    }

    idx2ids.clear();
}

void NNGKMeans::saveLogs(const char *dstFn)
{
    ofstream *outStrm = new ofstream(dstFn, ios::out);

    unsigned int i = 0;
    for(i = 0 ; i < nLogs; i++)
    {
       (*outStrm)<<i<<"\t\t"<<tmLogs[i]<<"\t\t"<<dstLogs[i]<<endl;
    }

    outStrm->close();
}

void NNGKMeans::freeOperation()
{
    if(this->arrayD != NULL)
    {
        free(arrayD);
        this->arrayD = NULL;
    }
    if(this->Ds != NULL)
    {
        free(Ds);
        Ds = NULL;
    }
    if(this->Ns != NULL)
    {
        free(this->Ns);
        this->Ns = NULL;
    }
    if(this->Es != NULL)
    {
        free(this->Es);
        this->Es = NULL;
    }
    if(this->topDist != NULL)
    {
        free(topDist);
        topDist = NULL;
    }
    if(!this->outNN&&this->topList != NULL)
    {
        free(topList);
        topList = NULL;
    }

    if(this->tmLogs != NULL)
    {
        delete [] tmLogs;
        tmLogs = NULL;
    }

    if(this->dstLogs != NULL)
    {
        delete [] dstLogs;
        dstLogs = NULL;
    }

    idx2ids.clear();
}
void NNGKMeans::test()
{
    const char *dstfn = "../../../data/cluster.txt";
    const char *srcfn = "../../../data/sift100k.txt";
    NNGKMeans *mykm = new NNGKMeans();
    int clusterNum = 128;
    mykm->buildcluster(srcfn, dstfn, "non", "large", "i2", clusterNum, false);
    delete mykm;

}



