// Two-dimensional line fitting and Coner Detector 
// by Sangkny for detecting the corner of round rectangle
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <io.h> // file structure

#include "utility.h"
#include "histogramvertical.h"
#include "histogramhorizontal.h"

using namespace std;
using namespace cv;
using namespace lpdr;

//#define CommandLine_Parameters

#define _sk_Memory_Leakag_Detector
#ifdef _sk_Memory_Leakag_Detector
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <vld.h>

#if _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#endif

const int ESCAPE_KEY = 27;
const std::string MAIN_WINDOW_NAME = "Corner Detector Main Window";

void help(char **argv) {
  cout << "\n Fast Corner Detector "
    << "\nCall"
    << "\n" << argv[0] << " file_path [top(0)/bottom(1)] [left-region(x y width height)] [right-region(x y width height)]" 
    << "\n\n (for example, CornerDetector_v100.exe C:/file_path/ 0 868 1230 300 300 2940 1230 300 300)"<<"\n"
		<< "\n 'q', 'Q' or ESC to quit"
		<< "\n" << endl;
}
struct cornerParameters {
  int min_distance_from_corner = 5;				// minimum distance from the corner point
  int max_distance_from_corner = 100;     // maximum distanc from the cornper point
  float corner_min_withheight_ratio = 0.95;
  float corner_max_withheight_ratio = 2.5;
  float offsetPerc = 0.10;		// offset % point from the center 0.065
  int adpthres_blockSize = 25;	// block size will determine the quality of the corner detection. + -> more edge but more robust edge but lost smooth changing edges
  double adpthres_C = 20;		// decreasing the value results in noisy and more detailed edge detection.
  bool disqualified = false;	// disqualified flag
  std::string reason = "";		// reason
  bool debugGeneral = true;		// debugging options belows
  bool debugShowImages = true;       // show image files
  bool debugShowCornerImages = true; // shows corner image or not
  bool debugShowImagesDetail = true; // can analyze the detail verson of code
} corParam;
// if you want to change the parameters, you can change the parameters as
//struct cornerParameters corParam;
//corParam


// sub function definition
/* this function returns a coner point from the given Grayscale ROI(region of interest) image.
roi_Gray  : grayscale roi image
roiIdx    : ROI Index for debugging
hs_leftright : horizontal searching direction for left (0) and right (1) directions from the center of Positive(Black)  Pole
vs_topbottom : vertical searching direction for top (0) and bottom (1) directions from the center of Positive (Black) Pole
*conP     : Parameter settings
--------------------------- hs/vs setting guide -> top2bottom and left2right direction
0. Top-Left Region      : false,  false;
1. TOP-Right Region     : true,   false;
2. Bottom-Right Region  : true,   true;
3. Bottom-Left Region   : false,  true;
*/
cv::Point findCorner(cv::Mat roi_Gray, int roiIdx, bool hs_leftright, bool vs_topbottom, cornerParameters * conP);

/* 
*/
bool verifyCornerPoint(cv::Mat &Binary, bool hs_leftright, bool vs_topbottom, lpdr::LineSegment &h_line, lpdr::LineSegment &v_line, cv::Point &cornerPt, cornerParameters * conP);

// sub function definition ends
int main(int argc, char **argv) {
#ifdef CommandLine_Parameters
	if (argc < 11) {
		help(argv);
		return 0;
	}
#endif
	
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	// roi selection 2 regions, 4 regions for both top and bottom locator	
	//main
	string input_file, full_file_path, folder_path;
	struct _finddata_t c_file;	// include <io.h>
#ifdef CommandLine_Parameters
	folder_path = argv[1];
#else
  /* office */
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/1-1/PASS/0004/0019/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/fail/top/";
  folder_path = "D:/sangkny/software/projects/2dLineFitting/data/fail/bottom/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/error/bottom/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/bottom/";
  //folder_path = "D:/sangkny/software/Projects/2dLineFitting/data/1-1/PASS/0004/temp/";
  /* home */
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/1-1/PASS/0004/0023/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/bottom/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/1-1/PASS/0004/temp/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/0023/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/fail/";
#endif // CommandLine_Parameters
	full_file_path = folder_path + "*.jpg";
	intptr_t hFile = _findfirst(full_file_path.c_str(), &c_file);
	if (hFile == -1) {
		std::cout << "No files in this folder!" << endl;
		return -2; // file error
	}

	Mat frame, frame_gray, crop_gray, dst;	// src, src_gray, cropped gray, destination images
	// roi detection is required, the following will be inserted into the do loop
	std::vector<Rect> roiRects;		// 2 roi selections: top-to-bottom and left-to-right direction	
	
#ifdef CommandLine_Parameters
  bool bTopBottomPlate = (bool)atoi(argv[2]);
  int lx, ly, lwidth, lheight;
  int rx, ry, rwidth, rheight;
  lx=atoi(argv[3]), ly=atoi(argv[4]), lwidth=atoi(argv[5]), lheight=atoi(argv[6]);
  rx=atoi(argv[7]), ry=atoi(argv[8]), rwidth=atoi(argv[9]), rheight=atoi(argv[10]);

  roiRects.push_back(Rect(lx, ly, lwidth, lheight));	// 0th  top-left
  roiRects.push_back(Rect(rx, ry, rwidth, rheight));  // 1st top-right
#else  
	//bool bTopBottomPlate = false;  // top Plate = false, bottom Plate = true;
	////roiRects.push_back(Rect(868, 1230, 300, 300));	// 0th  top-left
	//// roiRects.push_back(Rect(2940, 1230, 300, 300)); // 1st top-right  
	//roiRects.push_back(Rect(590, 952, 700, 700));	// 0th  top-left	
	//roiRects.push_back(Rect(2772, 952, 700, 700)); // 1st top-right  
  
	bool bTopBottomPlate = true;  // top Plate = false, bottom Plate = true;
	//roiRects.push_back(Rect(846, 1412, 300, 300));  // 2nd bottom-left
	//roiRects.push_back(Rect(2950, 1412, 300, 300)); // bottom-right	
	roiRects.push_back(Rect(548, 1366, 700, 700));  // 2nd bottom-left
	roiRects.push_back(Rect(2724, 1366, 700, 700)); // bottom-right	

#endif // CommandLine_Parameters
	//Mat element = getStructuringElement(MORPH_RECT, cv::Size(3,5));	// (3,3) 
	do {		
		destroyAllWindows();
		input_file = folder_path + string(c_file.name);
		frame = imread(input_file);
		if (frame.empty()) { cout << "File does not exist!! Please check the filename." << "\n";	waitKey(0); return -1; }
				// top-left roi first
		if (frame.channels() > 1)
			cvtColor(frame, frame_gray, CV_BGR2GRAY);
		else
			frame_gray = frame.clone();

		// process begins : assumption - gray input image, and I will refer the original address to reduce the processing time
		double t1 = (double)cvGetTickCount();		
		std::vector<cv::Point> cornerPts;
		for (int roiIdx = 0; roiIdx < 2; roiIdx++) {
			crop_gray = frame_gray(roiRects.at(roiIdx)); // top-left region and the processing starts from br.
			bool bRegion1, bRegion2;
			 if (!bTopBottomPlate) { // region selection from 4 corner regions
				if (roiIdx == 0) {
					bRegion1 = false;
					 bRegion2 = false;
				}
				else {
					bRegion1 = true;
					bRegion2 = false;
				}
			}
			else
			{
			if (roiIdx == 0) { // bottom left
				bRegion1 = false;
				bRegion2 = true;
			}
			else {            // bottom right
				bRegion1 = true;
				bRegion2 = true;
			}
			}
			Point cornerPt = findCorner(crop_gray, roiIdx, bRegion1, bRegion2, &corParam);
			if (cornerPt.x > 0 && cornerPt.y > 0)
			cornerPts.push_back(cornerPt);			
		}// roiIdx == 0 
		// process ends
		// print the processing time and file name
		double t2 = (double)cvGetTickCount();
		double t3 = (t2 - t1) / (double)getTickFrequency();

		cout << "Corner Detection time >>> " << t3 *1000. << "ms." << "\n";
		cout << string(c_file.name) << "\n";

		// show the result
		if (corParam.debugGeneral) {			
			if (corParam.debugShowImages) {
				Mat debugImg = frame.clone();
				if (frame.channels() < 3)
					cvtColor(frame, debugImg, CV_GRAY2BGR);
				for (int roiIdx = 0; roiIdx < 2; roiIdx++) {					
					rectangle(debugImg, roiRects[roiIdx], Scalar(0, 255, 0), 3, 8);
					if (cornerPts.size() > 1) {
						circle(debugImg, Point(roiRects[roiIdx].x, roiRects[roiIdx].y) + cornerPts[roiIdx], 9, Scalar(0, 0, 255), -1, 8);
					}
				}
				cv::namedWindow(MAIN_WINDOW_NAME, cv::WINDOW_NORMAL);
				cv::imshow(MAIN_WINDOW_NAME, debugImg);
				int showWindowWidth = 600;
				cv::resizeWindow(MAIN_WINDOW_NAME, showWindowWidth, (int)((float)frame.rows / ((float)frame.cols / (float)showWindowWidth)));
				//cout << " Window size is adjusted to the width of " << showWindowWidth << ", ratio : " << (float)frame.cols / (float)showWindowWidth << endl;
				cout << "Process is done." << endl;
			}
		}		

		int16_t c;
		c = waitKey(0);		
		if (c == ESCAPE_KEY/* (char) 27 ESC */ /*waitKey(50) == 'q' || waitKey(50) == 'Q'*/) break;
		frame.release(); dst.release();
	} while (_findnext(hFile, &c_file) == 0);// inside do for dealing with all the files in the given folder by sangkny
	_findclose(hFile);
	// main end
#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	
	return 0;
}

// linefitting and corner detector
cv::Point findCorner(cv::Mat roi_Gray, int roiIdx, bool hs_leftright, bool vs_topbottom, cornerParameters * conP) {
  /*vector<Mat> thresholds = produceThresholds(roi_Gray);
  for (int i = 0; i < thresholds.size(); i++) {
  displayImage("threshold : " + to_string(i), thresholds[i]);
  waitKey(1);
  }*/
  assert(roi_Gray.type() == CV_8UC1); // one channel unsigned char
  Mat binary, temp;
  LineSegment lineHor, lineVer;       // horizontal and vertical lines
  Point FinalCornerPoint(0, 0);        // a final corner point
  int offsetFromCenter = max(10, min(30, (int)((float)roi_Gray.cols * conP->offsetPerc)));
  int adaptiveThreshold_blockSize = conP->adpthres_blockSize;
  double adaptiveThreshold_C = conP->adpthres_C;
  /*
  // block 싸이즈를 키우면 강한 에지만를 잡을 수 있지만 놓치는 에지가 생길 수 있다. 하지만, 잡음은 없앨 수 있다.
  // threshold C (mean에서 빼주는 값)를 높이면 threshold 가 낮아지는 효과가 나기 때문에 잡음이 많고 세세한 부분까지 잡을 수 있음.
  // 마지막 term인 C를 많이 낮추면 opening을 해야 하는 수가 생김..
  // 300x300 의 경우는 25, 30 이 좋은 효과가 있음.
  // 700x700의 경우는 35, 또는 C를 20으로 주면 대부분의 에지를 잡음.
  */
  //adaptiveThreshold(roi_Gray, binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 30); 300x300
  adaptiveThreshold(roi_Gray, binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, adaptiveThreshold_blockSize/*25*/, adaptiveThreshold_C/*20*/);
  if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
    lpdr::displayImage("crop_gray " + to_string(roiIdx), roi_Gray);
    lpdr::displayImage("Binary " + to_string(roiIdx), binary);
  }

  Mat histMask = Mat::ones(binary.size(), binary.type());
  HistogramVertical vhist(binary, histMask);
  HistogramHorizontal hhist(binary, histMask);
  if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
    lpdr::displayImage("Ver histogram" + to_string(roiIdx), vhist.histoImg);
    lpdr::displayImage("Hor histogram" + to_string(roiIdx), hhist.histoImg);
  }
  int maxIdxx = vhist.getLocalMaximum(1, binary.cols - 1); // histogram maximum bin index which indicates the location of maximumu histogram peak.
  int maxIdxy = hhist.getLocalMaximum(1, binary.rows - 1);
  int maxValuex = vhist.getHeightAt(maxIdxx);
  int maxValuey = hhist.getHeightAt(maxIdxy);
  // make cases according to the hs_/vs_ conditions (4 cases)  
  int startX, endX, dX, startY, endY, dY;
  int line_sX, line_endX, line_sY, line_endY;

  /* Top-Left Region Search */
  if (hs_leftright == false && vs_topbottom == false)
  {
    // top-left search
    int xEdge = -1;
    int yRef = binary.rows - offsetFromCenter; // y fixed
    int peak_loc_variancex = 5; // 			
    for (int xi = binary.cols - 1; xi >= 0; xi--) {
      if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        line(tempImg, Point(0, yRef), Point(binary.cols - 1, yRef), Scalar(0, 255, 255), 1, 8); // yellow
        line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);		// red line
        imshow("Detected line (x,y) > 0 for vertical line searching", tempImg);
        cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
        cvWaitKey(1);
      }
      if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi)>0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi)>0) {
        xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
        break;
      }
    }
    if (xEdge == -1) {
      cout << "Edge is not found. Please check the ROIin the Top-Left Region!!" << endl;
    }
    else {
      // search the histogram peak and its length in Binary image or make a line      
      if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
        lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // 
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
        int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
        int maxValue1 = vhist.getHeightAt(maxIdx1);
        lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
        if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
    }
    // top-left field search	to finde a horizontal line			
    int yEdge = -1;
    int xRef;
    xRef = (xEdge == -1) ? binary.cols - offsetFromCenter : ((xEdge + (int)(float)((binary.cols - xEdge) + 1) / 2.)); // x fixed, from bottom to up direction
    int peak_loc_variancey = 5; // 			
    for (int yi = binary.rows - 1; yi >= 0; yi--) {
      //for (int yi = 0;yi< binary.rows - 1; yi++) { // sangkny dif
      if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        //tempImg = Mat::zeros(binary.size(), binary.type());
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
        line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
        line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 for horizontal line in the top-left region", tempImg);
        cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")" "-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
        cvWaitKey(1);
      }
      if ((int)binary.at<uchar>(yi, xRef) > 0 || /*(int)binary.at<uchar>(yi, max(0, min(binary.cols -1, xRef- offsetFromCenter))) > 0 ||*/ (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0) {
        yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
        break;
      }
    }
    if (yEdge == -1) {
      cout << "Edge is not found to y direction. Please check the ROI in the Top-Left Region !!" << endl;
    }
    else {
      // search the histogram peak and its length in Binary image or make a line      
      if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// vertical line in a range
        lineHor = LineSegment(Point(binary.cols - maxValuey, maxIdxy), Point(xRef, maxIdxy));
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
             // separate this region into two regions
        int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
        int maxValue1 = hhist.getHeightAt(maxIdx1);
        lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
        }
      }
    }

  }

  /* Top-Right region (it has left and right side) */
  if (hs_leftright == true && vs_topbottom == false)
  {
    // top-right field search
    int xEdge = -1;
    int yRef = binary.rows - offsetFromCenter; // y fixed
    int peak_loc_variancex = 5; // 			
    for (int xi = 0; xi< binary.cols; xi++) {        // dif for (int xi = binary.cols - 1; xi >= 0; xi--) {
      if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        line(tempImg, Point(0, yRef), Point(binary.cols - 1, yRef), Scalar(0, 255, 255), 1, 8);
        line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
        cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
        cvWaitKey(1);
      }
      if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
        xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
        break;
      }
    }
    if (xEdge == -1) {
      cout << "Edge is not found. Please check the ROI in Top-Right Region!!" << endl;
    }
    else {
      // search the histogram peak and its length in Binary image or make a line      
      if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
        lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // no changes line from top to bottom direction
        if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
        int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
        int maxValue1 = vhist.getHeightAt(maxIdx1);
        lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
        if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
    }
    // top-right and right side field search to detect a horizontal line
    int yEdge = -1;
    int xRef;
    xRef = (xEdge == -1) ? offsetFromCenter : min(offsetFromCenter, (xEdge + 1) / 2); // //sangkny dif x fixed, from bottom to up direction, left side was binary.cols - 10;
    int peak_loc_variancey = 5; // 			
    for (int yi = binary.rows - 1; yi >= 0; yi--) {
      //for (int yi = 0;yi< binary.rows - 1; yi++) {
      if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        //tempImg = Mat::zeros(binary.size(), binary.type());
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
        line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
        line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 for horizontal line in the top-right region", tempImg);
        cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
        cvWaitKey(1);
      }
      if ((int)binary.at<uchar>(yi, xRef) > 0 || (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 /*|| (int)binary.at<uchar>(yi,max(0, min(binary.cols-1,xRef+ offsetFromCenter))) > 0 */) {
        yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
        break;
      }
    }
    if (yEdge == -1) {
      cout << "Edge is not found to y direction. Please check the ROI in Top-Right Regin" << endl;
      // something wrong message
    }
    else {
      // search the histogram peak and its length in Binary image or make a line     
      if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
        lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
             // separate this region into two region
        int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
        int maxValue1 = hhist.getHeightAt(maxIdx1);
        lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
        }
      }
    }
  }

  /* Bottom-Left Region Search */
  if (hs_leftright == false && vs_topbottom == true) // Bottom-Left region
  {	  // bottom-left field search
    int xEdge = -1;
    int yRef = offsetFromCenter;		// y location fixed  // sangkny dif
    int peak_loc_variancex = 5; // 			
    for (int xi = binary.cols - 1; xi >= 0; xi--) { // dif different code region from other cases
      if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
        cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
        cvWaitKey(1);
      }
      if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
        xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
        break;
      }
    }
    if (xEdge == -1) {
      cout << "Edge is not found. Please check the ROI! in Bottom-Left Region" << endl;
    }
    else {
      // search the histogram peak and its length in Binary image or make a line		  
      if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
        lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // no changes line from top to bottom direction
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
        int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
        int maxValue1 = vhist.getHeightAt(maxIdx1);
        lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
    }
    // Detecting a horizontal line in the Bottom-Left Region
    int yEdge = -1;
    int xRef; // = binary.cols - offsetFromCenter; // //sangkny dif x fixed, from top-to-bottom(true) direction, the left side was binary.cols - 10;
    xRef = (xEdge == -1) ? binary.cols - offsetFromCenter : ((xEdge + (int)(float)((binary.cols - xEdge) + 1) / 2.)); // x fixed, from bottom to up direction
    int peak_loc_variancey = 5; // 			
                                //for (int yi = binary.rows - 1; yi >= 0; yi--) {
    for (int yi = 0; yi< binary.rows - 1; yi++) {			// sangkny dif 
      if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        //tempImg = Mat::zeros(binary.size(), binary.type());
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);			  
        line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
        line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 in the Bottom-Left Region", tempImg);
        cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
        cvWaitKey(1);
      }
      if ((int)binary.at<uchar>(yi, xRef) > 0 || /*(int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 ||*/ (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0) {
        yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
        break;
      }
    }
    if (yEdge == -1) {
      cout << "Edge is not found to search a horizontal line in the y direction. Please check the ROI!!" << endl;
      // something wrong message
    }
    else {
      // search the histogram peak and its length in Binary image or make a line
      if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
        lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
             // separate this region into two region
        int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
        int maxValue1 = hhist.getHeightAt(maxIdx1);
        lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
        }
      }
    }
  }

  /* Bottom-Right Region Search Algorithm */
  if (hs_leftright == true && vs_topbottom == true) // Bottom-Right Region Definition
  {
    // Bottom-left side,  a vertical line search
    int xEdge = -1;
    int yRef = offsetFromCenter;				// y fixed  : dif
    int peak_loc_variancex = 5;	// 			
    for (int xi = 0; xi< binary.cols; xi++) {        // dif for (int xi = binary.cols - 1; xi >= 0; xi--) {
      if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
        cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
        cvWaitKey(1);
      }
      if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
        xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
        break;
      }
    }
    if (xEdge == -1) {
      cout << "Edge is not found to detect a vertical line in the Bottom-Right Region. Please check the ROI !!" << endl;
      // disqualified
    }
    else {
      // search the histogram peak and its length in Binary image or make a line
      if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
        lineVer = LineSegment(Point(maxIdxx, yRef), Point(maxIdxx, maxValuex)); // diff no changes line from top to bottom direction
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
        int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
        int maxValue1 = vhist.getHeightAt(maxIdx1);
        lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
        }
      }
    }
    // In the Bottom-Right Region, and Right side field search to detect a horizontal line
    int yEdge = -1;
    int xRef; // = offsetFromCenter; // //sangkny dif x fixed, from bottom to up direction, left side was binary.cols - 10;
    xRef = (xEdge == -1) ? offsetFromCenter : min(offsetFromCenter, (xEdge + 1) / 2); // this code is more reasonable because the detected vertical line is robust and important
    int peak_loc_variancey = 5; // 			
                                //for (int yi = binary.rows - 1; yi >= 0; yi--) {
    for (int yi = 0; yi< binary.rows - 1; yi++) {					// sangkny dif
      if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
        Mat tempImg;
        tempImg = binary.clone();
        //tempImg = Mat::zeros(binary.size(), binary.type());
        if (tempImg.channels() < 3)
          cvtColor(tempImg, tempImg, CV_GRAY2BGR);
        //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
        line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
        line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
        imshow("Detected line (x,y) > 0 in the Bottom-Right Region", tempImg);
        cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
        cvWaitKey(1);
      }
      if ((int)binary.at<uchar>(yi, xRef) > 0 || (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 /*|| (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0*/) {
        yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
        break;
      }
    }
    if (yEdge == -1) {
      cout << "Edge is not found to detection a horizontal line in the y direction for Bottom-Right Region. Please check the ROI!!. Disqualified" << endl;
      // something wrong message  // disqualified
    }
    else {
      // search the histogram peak and its length in Binary image or make a line    
      if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
        lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
        }
      }
      else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
             // separate this region into two region
        int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
        int maxValue1 = hhist.getHeightAt(maxIdx1);
        lineHor = LineSegment(Point(xRef, maxIdx1), Point(binary.rows - maxValue1, maxIdx1));
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat debugTempImg = binary.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
          lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
        }
      }
    }
  }

  /*---> Corner Point Detection Condition and Detection <--*/
  if (lineHor.length > 0 && lineVer.length > 0) {
    // corner point
    Point corPt = lineHor.intersection(lineVer);
    // verify corner point
    bool bconfirmedCornerPt = verifyCornerPoint(binary, hs_leftright, vs_topbottom, lineHor, lineVer, corPt, conP);      
    
    if (bconfirmedCornerPt && corPt.x > 0 && corPt.x < binary.cols && corPt.y > 0 && corPt.y < binary.rows) { // more condition is required
      FinalCornerPoint = corPt;
      if (conP->debugGeneral) {
        cout << "Corner Point:(x,y) -> (" << FinalCornerPoint.x << ", " << FinalCornerPoint.y << ")." << endl;
        if (conP->debugShowCornerImages) {
          Mat debugTempImg = roi_Gray.clone();
          if (debugTempImg.channels() < 3)
            cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
          circle(debugTempImg, FinalCornerPoint, 3, Scalar(0, 0, 255), 2, 8);
          //displayImage("Corner Points:" + to_string(roiIdx), debugTempImg);
          std::string win_Name = "Corner Points:";
          cv::namedWindow(win_Name + to_string(roiIdx), cv::WINDOW_NORMAL);
          cv::imshow(win_Name + to_string(roiIdx), debugTempImg);
          int showWindowWidth = min(roi_Gray.cols, 300);
          cv::resizeWindow(win_Name + to_string(roiIdx), showWindowWidth, (int)((float)roi_Gray.rows / ((float)roi_Gray.cols / (float)(showWindowWidth))));
          cv::waitKey(1);
        }
      }
    }
    else {
      FinalCornerPoint = Point(0, 0);
      if (conP->debugGeneral) {
        if (roiIdx == 0)
          cout << "Left Region: Corner Vefirication is failed !!" << endl;
        else        
          cout << "Right Region: Corner Vefirication is failed !!" << endl;
      }
    }
  }

  return FinalCornerPoint;
}

bool verifyCornerPoint(Mat &Binary, bool hs_leftright, bool vs_topbottom, LineSegment &h_line, LineSegment &v_line, Point &cornerPt, cornerParameters *conP) {
  // case study according to the ROI region
  assert(Binary.type() == CV_8UC1); // should be binary (0, 255)
  int h_dist = -1, v_dist = -1; 
  int x_dist_Idx = -1, y_dist_Idx = -1;
  int sPtX = cornerPt.x;            // starting point (x,y)
  int sPtY = cornerPt.y; 
  int min_distance = conP->min_distance_from_corner;
  int max_distance = conP->max_distance_from_corner;
  float corner_MaxWidthHeight_Ratio = conP->corner_max_withheight_ratio;
  float corner_MinWidthHeight_Ratio = conP->corner_min_withheight_ratio;
  if (hs_leftright == false && vs_topbottom == false) {       // Top-Left ROI Region 
    //horizontal and vertical distance computation by turns
    // x-> +, y -> +
    for (int xi = sPtX; xi <= min(Binary.cols - 1, sPtX + max_distance); xi++) {
      if (Binary.at<uchar>(sPtY,xi) > 0) {
        x_dist_Idx = xi;
        break;
      }
    }
    if (x_dist_Idx == -1)
      return false;
    for (int yi = sPtY; yi <= min(Binary.rows - 1, sPtY + max_distance); yi++) {
      if (Binary.at<uchar>(yi, sPtX) > 0) {
        y_dist_Idx = yi;
        break;
      }
    }
    h_dist = abs(x_dist_Idx - sPtX);
    v_dist = abs(y_dist_Idx - sPtY);
    if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) && (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist)))
      return true;
  }
  else if (hs_leftright == true && vs_topbottom == false) {    // Top-Right ROI Region  
    // x-> -, y-> +
    for (int xi = sPtX; xi >= max(0, sPtX - max_distance); xi--) {
      if (Binary.at<uchar>(sPtY, xi) > 0) {
        x_dist_Idx = xi;
        break;
      }
    }
    if (x_dist_Idx == -1)
      return false;
    for (int yi = sPtY; yi <= min(Binary.rows - 1, sPtY + max_distance); yi++) {
      if (Binary.at<uchar>(yi, sPtX) > 0) {
        y_dist_Idx = yi;
        break;
      }
    }
    h_dist = abs(x_dist_Idx - sPtX);
    v_dist = abs(y_dist_Idx - sPtY);
    if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) && (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist)))
      return true;

  }
  else if (hs_leftright == false && vs_topbottom == true) {    // Bottom-Left ROI Region
    // x-> +, y -> -
    for (int xi = sPtX; xi <= min(Binary.cols - 1, sPtX + max_distance); xi++) {
      if (Binary.at<uchar>(sPtY, xi) > 0) {
        x_dist_Idx = xi;
        break;
      }
    }
    if (x_dist_Idx == -1)
      return false;
    for (int yi = sPtY; yi >= max(0, sPtY - max_distance); yi--) {
      if (Binary.at<uchar>(yi, sPtX) > 0) {
        y_dist_Idx = yi;
        break;
      }
    }
    h_dist = abs(x_dist_Idx - sPtX);
    v_dist = abs(y_dist_Idx - sPtY);
    if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) && (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist)))
      return true;
  }
  else if (hs_leftright == true && vs_topbottom == true){     // Bottom-Right ROI region
    // x-> -, y -> -
    for (int xi = sPtX; xi >= max(0, sPtX - max_distance); xi--) {
      if (Binary.at<uchar>(sPtY, xi) > 0) {
        x_dist_Idx = xi;
        break;
      }
    }
    if (x_dist_Idx == -1)
      return false;
    for (int yi = sPtY; yi >= max(0, sPtY - max_distance); yi--) {
      if (Binary.at<uchar>(yi, sPtX) > 0) {
        y_dist_Idx = yi;
        break;
      }
    }
    h_dist = abs(x_dist_Idx - sPtX);
    v_dist = abs(y_dist_Idx - sPtY);
    if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) && (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist)))
      return true;

  }
  else {
    assert(false);
  }  
  return false;
}