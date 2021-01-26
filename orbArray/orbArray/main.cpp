#define CVUI_IMPLEMENTATION

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/utility.hpp>
#include<string>
#include<opencv2/xfeatures2d.hpp>
#include <fstream>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include"imageManager.h"
#include <map>
#include <stdarg.h>
#include <opencv2/imgproc/imgproc.hpp>
#include"cvui.h"	
#define WINDOW_NAME	"Trackbar"
#define WINDOW_CHECKBOX "CHECKBOX"
#define WINDOW_WARNING "WARNINGMESSAGE"
#define WARNING_MESSAGE1 "The parameters couldn't perform the process"
#define IMAGE_MODE IMREAD_COLOR
#define NUMBER_IMAGE 12

using namespace std;
using namespace cv;

string wideImagePath = "C:/Users/asus/Desktop/ATLAS/matchingPics/atlasImages/wf.JPG";
string mfImagePath = "C:/Users/asus/Desktop/ATLAS/matchingPics/atlasImages/mf";

double fastThresholdValue = 0.1;
double mKeypointDividerCoeff = 60;
double downscaleCoeffValue = 3;
int mCropCoeff = 32;
double mAreaThreshold = 2.0;


void trackbarCVUI()
{

	cv::Mat* frame = new cv::Mat(500, 410, CV_8UC3);

	int* width = new int(300);

	int* x = new int(10);

	cvui::init(WINDOW_NAME);

	while (true) {
		// Fill the frame with a nice color
		*frame = cv::Scalar(49, 52, 49);

		// More customizations using options.
		unsigned int options = cvui::TRACKBAR_DISCRETE | cvui::TRACKBAR_HIDE_SEGMENT_LABELS;
		cvui::text(*frame, *x, 10, "Please describe downscale coefficient value");
		cvui::trackbar(*frame, *x, 40, *width, &downscaleCoeffValue, (double)1, (double)8, 1, "%.0Lf", options, (double)1);

		cvui::text(*frame, *x, 120, "Please describe the sensitivity rate to find result");
		cvui::trackbar(*frame, *x, 150, *width, &mAreaThreshold, (double)0.1, (double)20.0, 2, "%.1Lf", cvui::TRACKBAR_DISCRETE, (double)0.1);

		cvui::text(*frame, *x, 230, "Please describe fastThresholdValue to decide good matches");
		cvui::trackbar(*frame, *x, 260, *width, &fastThresholdValue, (double)0.0, (double)20.0, 2, "%.1Lf", cvui::TRACKBAR_DISCRETE, (double)0.1);

		cvui::text(*frame, *x, 340, "Please describe keypoints sampling ");
		cvui::trackbar(*frame, *x, 370, *width, &mKeypointDividerCoeff, (double)10, (double)500, 5, "%.0Lf", options, (double)5);

		cvui::text(*frame, *x, 450, "Press ESC to send entered data ");


		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();

		// Show everything on the screen
		cv::imshow(WINDOW_NAME, *frame);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			destroyWindow(WINDOW_NAME);
			break;
		}
	}

	delete frame;
	delete width;
	delete x;
}

void useCheckBox(bool* tempControl, string* showMessagetoBox, int widthVal) {


	cv::Mat* frameCheckBox = new cv::Mat(200, widthVal, CV_8UC3);
	*frameCheckBox = cv::Scalar(49, 52, 49);

	cvui::init(WINDOW_CHECKBOX);
	while (true)
	{
		cvui::text(*frameCheckBox, 90, 10, "Press ESC to send entered data ");
		cvui::checkbox(*frameCheckBox, 90, 50, *showMessagetoBox, tempControl);
		cvui::update();
		cv::imshow(WINDOW_CHECKBOX, *frameCheckBox);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			destroyWindow(WINDOW_CHECKBOX);
			break;
		}
	}
	delete frameCheckBox;
}

void warningBox(string warningMessage) {


	cv::Mat* warningBox = new cv::Mat(80, 400, CV_8UC3);
	*warningBox = cv::Scalar(49, 52, 49);

	cvui::init(WINDOW_WARNING);
	while (true)
	{
		cvui::text(*warningBox, 20, 20, warningMessage);
		cvui::update();
		cv::imshow(WINDOW_WARNING, *warningBox);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			destroyWindow(WINDOW_WARNING);
			break;
		}
	}
	delete warningBox;
}

void defaultParameters(int forIndex) {

	
	vector<vector<double> > defVal { { 4,	4,		0.7,	40	} ,	// 0
									{ 3,	3.2,	0.7,	50	} ,	// 1
									{ 4,	4,		1,		10	} ,	// 2
									{ 3,	4,		0.7,	50	} ,	// 3
									{ 4,	4,		1,		10	} ,	// 4
									{ 4,	4,		6,		50	} ,	// 5
									{ 4,	4,		0.9,	20	} ,	// 6
									{ 4,	4,		0.6,	50	} , // 7
									{ 4,	4,		0.9,	20	} , // 8
									{ 4,	4,		0.5,	50	} ,	// 9
									{ 4,	4,		0.9,	20	} ,	// 10
									{ 4,	4,		0.8,	40	} ,	// 11
									};

	 downscaleCoeffValue	= defVal[forIndex][0];
	 mAreaThreshold			= defVal[forIndex][1];
	 fastThresholdValue		= defVal[forIndex][2];
	 mKeypointDividerCoeff	= defVal[forIndex][3];

}

int main(int argc, const char* argv[])
{
	Mat wideImage = imread(wideImagePath, IMAGE_MODE);

	bool checked = true;
	bool askEveytime = true;
	bool useDefaultParameters = true;

	string* showMessage = new string("Render images one-by-one");
	useCheckBox(&checked, showMessage, 400);
	delete showMessage;

	imageManager imgManage;

	imgManage.sceneImg = wideImage;

	for (int i = 0; i < NUMBER_IMAGE; i++) {

		if (checked) {
			trackbarCVUI();
		}
		else if (i == 0) {
			string* askQ = new string("Do you want to tune parameters for every image parameter?");
			useCheckBox(&askEveytime, askQ, 700);
			delete askQ;
		}

		if (askEveytime && !checked) {
			trackbarCVUI();
		}
		else if (!checked && !askEveytime) {
			defaultParameters(i);
		}

		imgManage.cropImg(i, wideImage);

		imgManage.downscale(mfImagePath);

		imgManage.runMatcher();

		if (!(imgManage.matchesGMS).size()) {
			warningBox(WARNING_MESSAGE1);
			continue;
		}

		imgManage.getCorners();

		if (checked) {
			imgManage.drawResult();
			namedWindow("Result", WINDOW_NORMAL);
			cv::imshow("Result", imgManage.sceneImg);
			waitKey(0);
		}
		else {
			imgManage.drawResult();
			if (i == 11) {
				namedWindow("Result", WINDOW_NORMAL);
				cv::imshow("Result", imgManage.sceneImg);
				waitKey(0);
			}
			continue;
		}
	}

	return 0;
}
