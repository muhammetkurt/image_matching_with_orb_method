#ifndef imageManager_H
#define imageManager_H
#include <opencv2/opencv.hpp>
#include<string>
#include<string.h>
#include<iostream>
#include <opencv2/core/types.hpp>
#include<opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include<stddef.h>
#include<stdlib.h>
#include <opencv2/core.hpp>
#include <vector>

using namespace cv;
using namespace std;


class imageManager {
private:
	int areasX[12] = { 95, 	510, 	1015, 1500, 1860, 2355, 2950, 3425, 3920, 4160, 4650, 5150 };
	int areasY[12] = { 1945, 	1990, 	1855, 1700, 1885, 1850, 1850, 1820, 1830, 1775, 1765, 1775 };
	int areasWidth[12] = { 1600, 	1600, 	1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 800, 800 };
	int areasHeigh[12] = { 1000, 	1000, 	1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 500, 500 };

	Rect retROI;

	Mat downscaledImage;
	Mat croppedImg;

	vector<Point2f> scene_corners;

	vector<KeyPoint> refKeypointsMatcher;
	vector<KeyPoint> imgKeypointsMatcher;

	Mat refDescMatcher;
	Mat imgDescMatcher;

public:
	vector<DMatch>  matchesGMS;
	Mat sceneImg;
	int useIndexPic;

	//	class downScaleImage {
	void downscale(string mfImages);
	Mat getDownscaledImage();

	//	class CroppedImages {
	void cropImg(int i, Mat& imgToCrop);
	Mat getCroppedImg();

	//	class calcKeypoints {
	void calcOrb(Mat& imageToProcess, vector<KeyPoint>* orbKeypoints, Mat& orbDescMats);

	//	class Corners {
	void getCorners();
	vector<Point2f> getResultCorner();

	//	class drawRes {
	void drawResult();
	bool controlDraw();

	//	class Matcher {
	void runMatcher();
	vector<DMatch>	getMatches();
};

#endif