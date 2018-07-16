// Two-dimensional line fitting and Coner Detector 
// by Sangkny for detecting the corner of round rectangle
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <io.h> // file structure

#include"CornerDetectorAPI.h" // DLL Definition Header

using namespace std;
using namespace cv;
using namespace lpdr;

#define CommandLine_Parameters

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
const std::string MAIN_WINDOW_NAME = "Corner Detector Main Window for v1.03 DLL";

void help(char **argv) {
  cout << "\n Fast Corner Detector by sangkny "
    << "\n Call"
    << "\n" << argv[0] << " file_path [top(0)/bottom(1)] [left-region(x y width height)] [right-region(x y width height)]" 
    << "\n\n (for example, CornerDetector_v103.exe C:/file_path/ 0 868 1230 300 300 2940 1230 300 300)"<<"\n"
		<< "\n 'q', 'Q' or ESC to quit"
		<< "\n" << endl;
}

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
  folder_path = "D:/sangkny/software/projects/2dLineFitting/data/1-1/PASS/0004/0022/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/fail/top/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/fail/bottom/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/error/bottom/";
  //folder_path = "D:/sangkny/software/projects/2dLineFitting/data/bottom/";
  //folder_path = "D:/sangkny/software/Projects/2dLineFitting/data/1-1/PASS/0004/temp/";
  /* home */
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/1-1/PASS/0004/0023/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/bottom/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/temp/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/fail/bottom/";
	//folder_path = "D:/sangkny/work/software/2dLineFitting/data/error/more/";
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

  /* Parameter Settings */
  lpdr::cornerParameters corParam; // parameter settings: you can adjust the parameters here. Or you can use it as it is.  
  corParam.debugShowCornerImages = true;  
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
	bool bTopBottomPlate = false;  // top Plate = false, bottom Plate = true;
	//roiRects.push_back(Rect(868, 1230, 300, 300));	// 0th  top-left
	// roiRects.push_back(Rect(2940, 1230, 300, 300)); // 1st top-right  
	roiRects.push_back(Rect(590, 952, 700, 700));	// 0th  top-left	
	roiRects.push_back(Rect(2772, 952, 700, 700)); // 1st top-right  
  
	//bool bTopBottomPlate = true;  // top Plate = false, bottom Plate = true;
	////roiRects.push_back(Rect(846, 1412, 300, 300));  // 2nd bottom-left
	////roiRects.push_back(Rect(2950, 1412, 300, 300)); // bottom-right	
	//roiRects.push_back(Rect(548, 1366, 700, 700));  // 2nd bottom-left
	//roiRects.push_back(Rect(2724, 1366, 700, 700)); // bottom-right	

#endif // CommandLine_Parameters

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
			double tt1 = (double)cvGetTickCount();
			//Point cornerPt = findCorner(crop_gray, roiIdx, bRegion1, bRegion2, &corParam);
      Point cornerPt = CornerDetectorFuncs::findCornerAPI(crop_gray, roiIdx, bRegion1, bRegion2, &corParam);
			double tt2 = (double)cvGetTickCount();
			double tt3 = (tt2 - tt1) / (double)getTickFrequency();
			if(corParam.debugGeneral && corParam.debugTiming)
				cout << "Single Corner Detection time >>> " << tt3 *1000. << "ms." << "\n";
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

