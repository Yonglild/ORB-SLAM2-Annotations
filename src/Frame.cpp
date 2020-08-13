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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;   //在哪里赋值？
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

/**
 * @brief Copy constructor (Tracking 563 当前帧复制为上一帧(循环))
 *
 * 复制构造函数, mLastFrame = Frame(mCurrentFrame)
 */
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];   //为什么要赋值？

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);  //设置相机姿态，即设置相机坐标系到世界坐标系的变换矩阵，世界坐标系到相机坐标系的变换矩阵
}


// 双目的初始化
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mb(0), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info   金字塔信息，为下一步提特征做准备
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 同时对左右目提特征 开启两个线程
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();   //提取的特征点个数

    if(mvKeys.empty())
        return;
    // Undistort特征点，这里没有对双目进行校正，因为要求输入的图像已经进行极线校正
    UndistortKeyPoints();

    // 计算双目间的匹配, 匹配成功的特征点会计算其深度
    // 深度存放在mvuRight 和 mvDepth 中
    ComputeStereoMatches();

    // 对应的mappoints
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));   
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// RGBD初始化
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();   //图像提金字塔的层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    //图像提金字塔的尺度因子
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();   //???
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();  //???
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    //mvKeys-->mvKeysUn
    UndistortKeyPoints();       // 调用OpenCV的矫正函数矫正orb提取的特征点

    //深度存储在mvDepth
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        //矫正边界点
        ComputeImageBounds(imGray);

        // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;   //基线

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    //分配特征点到网格
    AssignFeaturesToGrid();
}

// 单目初始化
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    // 调用OpenCV的矫正函数矫正orb提取的特征点
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


//125
    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    // 把每个特征点分配到对应网格 ，目的：加速匹配
void Frame::AssignFeaturesToGrid()
{
    //1.初始化网格空间
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);  //？？所有特征点均匀分配到每个网格*0.5
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);    //mGrid[][]里每一个元素都是nReserve维的向量vector<std::size_t>

    //2.把每个特征点分配到对应网格
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))   //得出特征点在哪个网格
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//84 146
//具体实现见ORBextractor.cpp
//提取ORB特征点及描述子
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

/** Frame 61
 * @brief Set the camera pose.
 * 
 * 设置相机姿态，即设置相机坐标系到世界坐标系的变换矩阵，世界坐标系到相机坐标系的变换矩阵
 * @param Tcw Transformation from world to camera
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();  //根据Tcw计算mRcw、mtcw和mRwc、mOw
}

/**289
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * 根据Tcw计算mRcw、mtcw和mRwc、mOw
 */
void Frame::UpdatePoseMatrices()
{
    // [x_camera 1] = [R|t]*[x_world 1]，坐标为齐次形式
    // x_camera = R*x_world + t
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mOw, 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系
    mOw = -mRcw.t()*mtcw;
}

/**
 * @brief 判断一个点是否在视野内，则该点可用作投影用Tracking 1507
 *
 * 计算了重投影坐标，观测方向夹角，预测在当前帧的尺度
 * @param  pMP             MapPoint
 * @param  viewingCosLimit 视角和平均视角的方向阈值
 * @return                 true if is in view
 * @see SearchLocalPoints()
 * 取出MapPoint点投影到相机像素坐标，判断是否越界；计算MapPoint到相机中心的距离, 并判断是否在距离范围内
 * 计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回；根据深度预测尺度（对应特征点在一层）
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // 3D点P在相机坐标系下的坐标
    const cv::Mat Pc = mRcw*P+mtcw; // 这里的Rt是经过初步的优化后的
    const float &PcX = Pc.at<float>(0);  //表示引用，它是变量Pc.at<float>(0)的别名.并非定义了一个新的变量
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // V-D 1) 将MapPoint投影到当前帧, 并判断是否在图像内
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // V-D 3) 计算MapPoint到相机中心的距离, 并判断是否在距离范围内
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // PO 相机到3D点P的向量，向量方向由相机指向3D点P
    const cv::Mat PO = P-mOw;    //mOw=mtwc
    const float dist = cv::norm(PO);    //相机到3d点P的距离

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // V-D 2) 计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回
    cv::Mat Pn = pMP->GetNormal();   //得到MapPoint的平均观测方向

    const float viewCos = PO.dot(Pn)/dist;    //dot点乘

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征点在一层）  根据dist返回ＭapPoint在pKF中的层数
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    // 该点在该帧的视野范围内，标记该点将来要被投影
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;     //该MapPoint在该帧的像素坐标
    pMP->mTrackProjXR = u - mbf*invz; //该3D点投影到双目右侧相机上的横坐标
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos = viewCos; //余弦角

    return true;
}

/**这个函数作用是什么？？
 * @brief 得到在以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]被提取的特征点的个数
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小层数（金字塔）
 * @param maxLevel 最大层数
 * @return         满足条件的特征点的序号
 */

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);   //预留N个长度
    //判断要求区域是否在图像内
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    //判断以x,y为中心,边长为2r的方形区域是否在图像内
    //遍历在方形区域内的方格子
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())   //该方格内没有特征点
                continue;
            //遍历该方格内的特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                //判断该点是否在[minLevel,maxLevel]被检测到
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                //矫正后的点是否仍在矫正前点的2r范围内
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

//262
//判断特征点在那个方格内
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief Bag of Words Representation
 *
 * 计算词包mBowVec和mFeatVec，其中mFeatVec记录了属于第i个node（在第4层）的ni个描述子
 * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
 */
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

//94 153
// 调用OpenCV的矫正函数矫正orb提取的特征点
void Frame::UndistortKeyPoints()
{
    //畸变系数为0，说明没有失真
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    //将该帧的所有特征点保存在N*2的mat中
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 调整mat的通道为2，矩阵的行列形状不变
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK); // 用cv的函数进行失真校正
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // 存储校正后的特征点mvKeysUn
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

//矫正图像畸变，确定新的图像边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{   //畸变系数不为0,说明需要矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 矫正前四个边界点：(0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0;         //左上
		mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; //右上
		mat.at<float>(1,1)=0.0;
		mat.at<float>(2,0)=0.0;         //左下
		mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; //右下
		mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//左下和右下纵坐标最小的
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**98
 * @brief 双目匹配!!!
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;  //描述子距离阈值

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;  //第0层金字塔的行数

    //Assign keypoints to row table
    // 步骤1：建立特征点搜索范围对应表，一个特征点在一个带状区域内搜索匹配特征点
    // 匹配搜索的时候，不仅仅是在一条横线上搜索，而是在一条横向搜索带上搜索,简而言之，原本每个特征点的纵坐标为1，这里把特征点体积放大，纵坐标占好几行
    // 例如左目图像某个特征点的纵坐标为20，那么在右侧图像上搜索时是在纵坐标为18到22这条带上搜索，搜索带宽度为正负2，搜索带的宽度和特征点所在金字塔层数有关
    // 简单来说，如果纵坐标是20，特征点在图像第20行，那么认为18 19 20 21 22行都有这个特征点
    // vRowIndices[18]、vRowIndices[19]、vRowIndices[20]、vRowIndices[21]、vRowIndices[22]都有这个特征点编号
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();  //原始右图像特征点数（未矫正）

    for(int iR=0; iR<Nr; iR++)
    {
        // !!在这个函数中没有对双目进行校正，双目校正是在外层程序中实现的
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大
        // 如果特征点在金字塔第一层，则搜索范围为:正负2
        // 尺度越大其位置不确定性越高，所以其搜索半径越大
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];  //搜索范围r
        const int maxr = ceil(kpY+r);  //向上取整
        const int minr = floor(kpY-r);  //向下取整

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);  //将特征点的索引压入？
    }

    // Set limits for search
    const float minZ = mb;        // NOTE bug mb没有初始化，mb的赋值在构造函数中放在ComputeStereoMatches函数的后面
    const float minD = 0;        // 最小视差, 设置为0即可
    const float maxD = mbf/minZ;  // 最大视差, 对应最小深度 mbf/minZ = mbf/mb = mbf/(mbf/fx) = fx (wubo???)

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;  //左目iL点的最小匹配误差
    vDistIdx.reserve(N);

    // 步骤2：对左目相机每个特征点，通过描述子在右目带状搜索区域找到匹配点, 再通过SAD做亚像素匹配
    // 注意：这里是校正前的mvKeys，而不是校正后的mvKeysUn
    // KeyFrame::UnprojectStereo和Frame::UnprojectStereo函数中不一致
    // 这里是不是应该对校正后特征点求深度呢？(wubo???)
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];  //左目原始图像上的特征点
        const int &levelL = kpL.octave;  //该点的金字塔层数
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];  //第VL层的全部特征点作为候选

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD; // 最小匹配范围？？
        const float maxU = uL-minD; // 最大匹配范围？？

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // 每个特征点描述子占一行，建立一个指针指向iL特征点对应的描述子
        const cv::Mat &dL = mDescriptors.row(iL);    //索引为iL特征点的描述子

        // Compare descriptor to right keypoints
        // 步骤2.1：遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小）
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];   //原始图像上的特征点

            // 仅对近邻尺度的特征点进行匹配（上下两层）
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            // 横向搜索范围[minU,maxU]
            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);  //右目索引为iR特征点的描述子
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);
                //最好的匹配的匹配误差存在bestDist，匹配点位置存在bestIdxR中
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 步骤2.2：通过SAD匹配提高像素匹配修正量bestincR（像素的灰度不变性假设）
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            //左右目匹配上的点可能不在同一金字塔层
            // kpL.pt.x对应金字塔最底层坐标，将最佳匹配的特征点对尺度变换到尺度对应层 (scaleduL, scaledvL) (scaleduR0)
            const float uR0 = mvKeysRight[bestIdxR].pt.x;  //原始图像上该特征点的横坐标（第0层？）
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);  //该点在对应层的横坐标
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);  //右目对应点在对应层的横坐标

            // sliding window search
            const int w = 5; // 滑动窗口的大小11*11 注意该窗口取自resize后的图像
            //左目对应层的窗口图像Mat,不会发生越界问题吗？？？
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);  //CV_32F 灰度值0-1
            IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);//窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

            int bestDist = INT_MAX;  //??
            int bestincR = 0;
            const int L = 5;  //滑动半径
            vector<float> vDists;  //记录滑动过程中Dist的大小
            vDists.resize(2*L+1);  // 11

            // 滑动窗口的横向滑动范围为（-L, L）
            const float iniu = scaleduR0+L-w; //这个地方是否应该是scaleduR0-L-w (wubo???)
            const float endu = scaleduR0+L+w+1;
            //若滑动范围越界，则退出
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 右目横向滑动窗口 范围[-L,L]
            for(int incR=-L; incR<=+L; incR++)
            {   //右目对应层窗口图像
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);//窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

                float dist = cv::norm(IL,IR,cv::NORM_L1); // 一范数，计算差的绝对值（光度误差？？)
                if(dist<bestDist)
                {
                    bestDist =  dist;// SAD匹配目前最小匹配偏差
                    bestincR = incR; // SAD匹配目前最佳的修正量
                }

                vDists[L+incR] = dist; // 正常情况下，这里面的数据应该以抛物线形式变化（大-->小-->大）
            }

            if(bestincR==-L || bestincR==L) //整个滑动窗口过程中，SAD最小值不是以抛物线形式出现，SAD匹配失败，同时放弃求该特征点的深度
                continue;

            // Sub-pixel match (Parabola fitting)
            // 步骤2.3：做抛物线拟合找谷底得到亚像素匹配deltaR
            // (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist)三个点拟合出抛物线
            // bestincR+deltaR就是抛物线谷底的位置，相对SAD匹配出的最小值bestincR的修正量为deltaR
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];
            //什么拟合法？
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 抛物线拟合得到的修正量不能超过一个像素，否则放弃求该特征点的深度
            if(deltaR<-1 || deltaR>1)  //亚像素修正量
                continue;

            // Re-scaled coordinate
            // 通过描述子匹配得到匹配点位置为scaleduR0
            // 通过SAD匹配找到修正量bestincR
            // 通过抛物线拟合找到亚像素修正量deltaR
            //得到右目最好点的亚像素横坐标
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // 这里是disparity，根据它算出depth
            float disparity = (uL-bestuR);  //视差

            if(disparity>=minD && disparity<maxD) // 最后判断视差是否在范围内
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // depth 是在这里计算的
                // depth=baseline*fx/disparity
                mvDepth[iL]=mbf/disparity;   // 左目iL点的深度
                mvuRight[iL] = bestuR;       // 匹配在右图的横坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL)); // 该特征点SAD匹配最小匹配偏差
            }
        }
    }

    // 步骤3：剔除SAD匹配偏差较大的匹配特征点
    // 前面SAD匹配只判断滑动窗口中是否有局部最小值，这里通过对比剔除SAD匹配偏差比较大的特征点的深度
    sort(vDistIdx.begin(),vDistIdx.end()); // 根据所有匹配对的SAD偏差进行排序, 距离由小到大
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median; // 计算自适应距离, 大于此距离的匹配对将剔除

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

//156
//RGBD从深度图imDepth中获取对应深度
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    // mvDepth直接由depth图像读取
    mvuRight = vector<float>(N,-1);      // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标
    mvDepth = vector<float>(N,-1);
    //遍历特征点
    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];      //经过矫正之后的特征点

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;    //即ur;  z=b*f/(ul-ur);  视差=ul-ur;
        }
    }
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
cv::Mat Frame::UnprojectStereo(const int &i)
{
    // KeyFrame::UnprojectStereo
    // mvDepth是在ComputeStereoMatches函数中求取的
    // mvDepth对应的校正前的特征点，可这里却是对校正后特征点反投影
    // KeyFrame::UnprojectStereo中是对校正前的特征点mvKeys反投影
    // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
