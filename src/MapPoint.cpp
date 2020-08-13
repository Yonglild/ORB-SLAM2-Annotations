/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

//第一种构造方式：给定点位置和关键帧。
/**
 * @brief 给定坐标与keyframe构造MapPoint
 *
 * 双目：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()
 * 单目：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pRefKF KeyFrame
 * @param pMap   Map
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    // 该MapPoint平均观测方向（按：是相机的观测方向吗？？？？）32位浮点数
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}


//第二种构造方式：给定坐标与frame构造MapPoint
/**
 * @brief 给定坐标与frame构造MapPoint
 *
 * 双目：UpdateLastFrame()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pMap   Map
 * @param pFrame Frame
 * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;// 初始化：世界坐标系下该帧到3D点的向量（相机-->MapPoint）
    mNormalVector = mNormalVector/cv::norm(mNormalVector);// 世界坐标系下相机到3D点的归一化

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);  //MapPoint到相机的距离
    const int level = pFrame->mvKeysUn[idxF].octave;   //该点从那一层提取。这里取金字塔中能够提取出该特征点最高层级作为该特征点的层级
    const float levelScaleFactor =  pFrame->mvScaleFactors[level]; //该层的金字塔缩放系数
    const int nLevels = pFrame->mnScaleLevels;    //图像提取金字塔的层数

    // 另见PredictScale函数前的注释
    mfMaxDistance = dist*levelScaleFactor;   //最大尺度的距离dmax
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];  //最小尺度的距离dmin

    // 见mDescriptor在MapPoint.h中的注释
    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);   //描述子按行排列

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}
    //加载地图用的
    MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap):
            mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1), mnFound(1), mbBad(false),
            mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3,1,CV_32F);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId=nNextId++;
    }

//设置p点世界坐标
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();   //MapPoint平均观测方向（所有帧到该点的向量的平均）
}

//返回参考关键帧
KeyFrame* MapPoint::GetReferenceKeyFrame()
{
     unique_lock<mutex> lock(mMutexFeatures);
     return mpRefKF;
}

/**
 *
 * @brief 添加观测（指被关键帧看到的次数）
 *
 * 记录哪些KeyFrame的那个特征点对应到该MapPoint \n
 * 并增加观测的相机数目nObs，单目+1，双目或者grbd+2
 * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些MapPoints的关键帧是共视关键帧
 * map<KeyFrame* ,size_t> mObservations很重要！！！
 * @param pKF KeyFrame
 * @param idx MapPoint在KeyFrame中的索引
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))   //count计数。如果有该参考关键帧，则返回
        return;
    // 记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
    mObservations[pKF]=idx;

    if(pKF->mvuRight[idx]>=0)
        nObs+=2; // 双目或者grbd //nObs是观测到地图点的相机数，即观测次数
    else
        nObs++; // 单目
}

//什么情况下会使用这个函数？？  当该点不出现在该帧时，则删除他们之间的关系
//KeyFrame.cpp 584  当关键帧pKF要被删除时，关键帧上的点擦除与pKF的关系
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];   //该点在pKF中的索引
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;
            //删除地图点中维护关键帧观测关系
            mObservations.erase(pKF);

            // 如果该keyFrame是参考帧，该Frame被删除后重新指定RefFrame
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;//?????

            // If only 2 observations or less, discard point
            // 当观测到该点的相机数目少于2时，丢弃该点
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag(); //分别删除mObservations,关键帧对应的点，和地图中对应的点。

}

//获得观测
map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

//返回该点被关键帧（KeyFrame）观测到的次数（帧个数）
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

//分别删除mObservations,关键帧对应的点，和地图中对应的点。
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;// 把mObservations转存到obs，obs和mObservations里存的是指针，赋值过程为浅拷贝
        mObservations.clear();// 把mObservations指向的内存释放，obs作为局部变量之后自动删除
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);//删除该关键帧索引对应的MapPoint.
    }

    mpMap->EraseMapPoint(this);// 从地图中删除MapPoint.擦除该MapPoint申请的内存
}

//返回替换后的点
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

////将当前地图点（this）,替换成pMp。LocalMapping SearchInNeighbors(),LoopClosing闭环中 满足什么条件之后才能替换？？？
// 在形成闭环的时候，会更新KeyFrame与MapPoint之间的关系
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;// 这一段和SetBadFlag函数相同
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();  //指针内存释放
        mbBad=true;//如果被替换,当前地图点的mbBad会被记为true
        nvisible = mnVisible;  //在视野范围内
        nfound = mnFound;    //能找到该旧点的帧数
        mpReplaced = pMP;
    }

    // 遍历所有能观测到旧MapPoint的keyframe，keyframe中对应的MapPoint都要替换成pMP
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        //如果新ＭapPoint点的mObservation中没有pKF，即新ＭapPoint没被pKF记录到，则互相添加连接
        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);//用pMP替代原先的该关键帧中的对应的MapPoint
            pMP->AddObservation(pKF,mit->second);// 添加关键帧和索引
        }
        else
        {   // 新的ＭapPoint也在pKF中，说明有冲突。更相信新的ＭapPoint！
            // 产生冲突，即pKF中有两个特征点a,b（这两个特征点的描述子是近似相同的），这两个特征点对应两个MapPoint为this,pMP
            // 然而在fuse的过程中pMP的观测更多，需要替换this，因此保留b与pMP的联系，去掉a与this的联系
            pKF->EraseMapPointMatch(mit->second);  //删除关键帧里对应的索引的MapPoint
        }
    }
    pMP->IncreaseFound(nfound);  //新的=旧的+新的？？？
    pMP->IncreaseVisible(nvisible); //？？
    pMP->ComputeDistinctiveDescriptors();  //计算新点具有代表性的描述子

    //Ｍap中删掉旧点
    mpMap->EraseMapPoint(this);
}

// 没有经过MapPointCulling检测的MapPoints
bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

/**
 * @brief Increase Visible
 *
 * Visible表示：
 * 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断(满足5个条件）
 * 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
 *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
 *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
 */
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

/**
 * @brief Increase Found
 *
 * 能找到该点的帧数+n，n默认为1
 * @see Tracking::TrackLocalMap()
 */
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}
//visible和found并不是等价的，found的地图点一定是visible的，
//但是visible的地图点可能not found
    //GetFoundRatio低表示该地图点在很多关键帧的视野内，
    //但是没有匹配上很多的特征点
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

/**268
 * @brief 计算具有代表的描述子
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
 * @see III - C3.3
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // 获得这些描述子两两之间的距离
    const size_t N = vDescriptors.size();
	
    //float Distances[N][N];
	std::vector<std::vector<float> > Distances;
	Distances.resize(N, vector<float>(N, 0));
	for (size_t i = 0; i<N; i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // 第i个描述子到其它所有所有描述子之间的距离
        //vector<int> vDists(Distances[i],Distances[i]+N);
		vector<int> vDists(Distances[i].begin(), Distances[i].end());
		sort(vDists.begin(), vDists.end());

        // 获得中值
        int median = vDists[0.5*(N-1)];
        
        // 寻找最小的中值
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        
        // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
        // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
        // 最好的描述子就是和其它描述子的平均距离最小
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

//返回该MapPoint在该关键帧的索引
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

/**
 * @brief check MapPoint is in keyframe
 * @param  pKF KeyFrame
 * @return     true if in pKF
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

/**
 * @brief 更新平均观测方向以及观测距离范围
 *
 * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量
 * @see III - C2.2 c2.4
 */
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations=mObservations; // 获得观测到该3d点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧
        Pos = mWorldPos.clone();    // 3d点在世界坐标系中的位置
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali); // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
        n++;
    } 

    cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
    const float dist = cv::norm(PC); // 该点到参考关键帧相机的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
        // 另见PredictScale函数前的注释
        mfMaxDistance = dist*levelScaleFactor;                           // 参考关键帧观测到该点的距离下限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 观测到该点的距离上限
        mNormalVector = normal/n;                                        // 获得平均的观测方向（所有帧到该点的向量的平均）
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)   向上取整层数
//            log(1.2)

//返回ＭapPoint在pKF中的层数
//Frame.cpp 372
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }

    // 同时取log线性化
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

//Frame.cpp 372
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
