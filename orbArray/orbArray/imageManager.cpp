#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>
#include <opencv2/core/types.hpp>
#include<opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include"imageManager.h"
#define IMAGE_MODE_INFO IMREAD_COLOR

using namespace std;
using namespace cv;

/*
 cromImg method is assigning interested area coordinates from predefined array to Rect variable.
After that, it crops the area from the image which is taken from main.cpp.
*/
void imageManager::cropImg(int i, Mat& imgToCrop) {
	retROI.x = areasX[i];
	retROI.y = areasY[i];
	retROI.width = areasWidth[i];
	retROI.height = areasHeigh[i];

	croppedImg = imgToCrop(retROI);
	useIndexPic = i;
}

// This method can be used from another cpp however we didn't use this.
Mat imageManager::getCroppedImg() {
	return croppedImg;
}

/*
This method takes string variable from source file and loads the image with string.
Then, I took downscaleCoeffValue which indicates how many times we need to downscale the reference image.
The final result of this method is given in private object downscaledImage.
*/
void imageManager::downscale(string mfImages) {

	extern double downscaleCoeffValue;

	string str2 = to_string(useIndexPic);
	string zeroStr = "0";
	if (useIndexPic < 10) {
		downscaledImage = imread(mfImages + zeroStr + str2 + ".JPG", IMAGE_MODE_INFO);
	}
	else {
		downscaledImage = imread(mfImages + str2 + ".JPG", IMAGE_MODE_INFO);
	}

	for (int j = 0; j < downscaleCoeffValue; j++)
	{
		pyrDown(downscaledImage, downscaledImage, Size(downscaledImage.cols / 2, downscaledImage.rows / 2));
	}
}

// This method is defined to take output of downscale method from out source such as main.cpp
Mat imageManager::getDownscaledImage() {

	return downscaledImage;
}

/*
calcOrb method is used for calculating keypoints and descriptor matrix of given image such as reference or cropped image.
Actually, we won't need to get this method output because this method is not going to perform in other source except imageManager Class.
*/
void imageManager::calcOrb(Mat& imageToProcess, vector<KeyPoint>* orbKeypoints, Mat& orbDescMats) {

	extern double mKeypointDividerCoeff;
	extern double	fastThresholdValue;

	Ptr<Feature2D> objORBDe = ORB::create((imageToProcess.cols * imageToProcess.rows) / mKeypointDividerCoeff);
	objORBDe.dynamicCast<ORB>()->setFastThreshold(fastThresholdValue);
	objORBDe->detectAndCompute(imageToProcess, noArray(), *orbKeypoints, orbDescMats);

}


/*
I called calcOrb method in runMatcher method to match feature of reference and cropped image.
So, match and xfeatures2d::matchGMS functions perform matching algorithm between reference and cropped image feature.
I changed some default parameter to obtain better result.
*/
void imageManager::runMatcher() {

	vector<DMatch>  matchesAll;

	calcOrb(croppedImg, &imgKeypointsMatcher, imgDescMatcher);

	calcOrb(downscaledImage, &refKeypointsMatcher, refDescMatcher);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	matcher->match(refDescMatcher, imgDescMatcher, matchesAll);
	xfeatures2d::matchGMS(downscaledImage.size(), croppedImg.size(), refKeypointsMatcher, imgKeypointsMatcher, matchesAll, matchesGMS, false, true, 6.);

}

/*
This method helps us to get result of runMatcher.
*/
vector<DMatch>	imageManager::getMatches() {
	return matchesGMS;
}

/*
getCorners method eliminates good features from bad ones and assign them into obj and scene vector. 
This assigning process is made by considering outputs of runMatcher method.
findHomography function's aim is to find homography matrix to correct scene perspective.
After finding homography result, I generate the final coordinates with perspectiveTransform function.
*/
void imageManager::getCorners() {

	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int k = 0; k < matchesGMS.size(); k++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(refKeypointsMatcher[matchesGMS[k].queryIdx].pt);
		scene.push_back(imgKeypointsMatcher[matchesGMS[k].trainIdx].pt);
	}

	Mat tHomography = findHomography(obj, scene, RANSAC);

	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(downscaledImage.cols, 0);
	obj_corners[2] = Point(downscaledImage.cols, downscaledImage.rows);
	obj_corners[3] = Point(0, downscaledImage.rows);

	perspectiveTransform(obj_corners, scene_corners, tHomography);

}

/*
If we want to take the output of getCorners method from out source, we can use this method (getResultCorner)
*/
vector<Point2f> imageManager::getResultCorner() {
	return scene_corners;
}


/*
This method helps us to draw lines on result image.
useIndexPic is current index and we use it to find rectangle points to fix result area.
controlDraw function compares result and our input to make sure there is no big difference between them.
*/
void imageManager::drawResult() {

	retROI.x = areasX[useIndexPic];
	retROI.y = areasY[useIndexPic];
	retROI.width = areasWidth[useIndexPic];
	retROI.height = areasHeigh[useIndexPic];

	if (controlDraw())
	{
		Point2f tWideAnglePoint(retROI.x, retROI.y);
		line(sceneImg, tWideAnglePoint + scene_corners[0], tWideAnglePoint + scene_corners[1], Scalar(0, 255, 255), 7);
		line(sceneImg, tWideAnglePoint + scene_corners[1], tWideAnglePoint + scene_corners[2], Scalar(0, 255, 0), 4);
		line(sceneImg, tWideAnglePoint + scene_corners[2], tWideAnglePoint + scene_corners[3], Scalar(255, 0, 0), 4);
		line(sceneImg, tWideAnglePoint + scene_corners[3], tWideAnglePoint + scene_corners[0], Scalar(255, 255, 0), 14);

		putText(sceneImg, to_string(useIndexPic), tWideAnglePoint + scene_corners[0], FONT_HERSHEY_SIMPLEX, 3, Scalar(255), 6);

	}
}

bool imageManager::controlDraw() {

	extern double downscaleCoeffValue;
	extern int mCropCoeff;
	extern double mAreaThreshold;

	float tDownCropRatio = float(downscaleCoeffValue) / float(mCropCoeff);

	//Create 2 Rectangles and compare it's areas
	vector<float> tSizeList;
	int tW = scene_corners[2].x - scene_corners[0].x;
	int tH = scene_corners[2].y - scene_corners[0].y;
	tSizeList.push_back(tW * tH);

	tW = scene_corners[1].x - scene_corners[3].x;
	tH = scene_corners[3].y - scene_corners[1].y;
	tSizeList.push_back(tW * tH);


	float dividePar = downscaledImage.rows * downscaledImage.cols;

	float tObjSceneRatio = (dividePar) / tSizeList.front();
	if (tObjSceneRatio < tDownCropRatio + mAreaThreshold && tObjSceneRatio > tDownCropRatio - mAreaThreshold)
	{
		tObjSceneRatio = (dividePar) / tSizeList.back();
		if (tObjSceneRatio < tDownCropRatio + mAreaThreshold && tObjSceneRatio > tDownCropRatio - mAreaThreshold)
		{
			return true;
		}
	}
	return false;
}
