// Two-dimensional line fitting Example
// by Sangkny for detecting the corner of round rectangle
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace std;

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

void help(char **argv) {
	cout << "\n 2D line fitting Test.."
		<< "\nCall"
		<< "\n" << argv[0] << "\n"
		<< "\n 'q', 'Q' or ESC to quit"
		<< "\n" << endl;
}

int main(int argc, char **argv) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif
	cv::Mat img(500, 500, CV_8UC3);
	cv::RNG rng(-1);
	help(argv);
	for (;;) {
		char key;
		int i, count = rng.uniform(0, 1000) + 3, outliers = count / 5;
		float a = (float)rng.uniform(0., 200.);
		float b = (float)rng.uniform(0., 40.);
		float angle = (float)rng.uniform(0., CV_PI);
		float cos_a = cos(angle), sin_a = sin(angle);
		cv::Point pt1, pt2;
		vector<cv::Point> points(count);
		cv::Vec4f line;
		float d, t;
		b = MIN(a * 0.3f, b);

		// generate some points that are close to the line
		for (i = 0; i < count - outliers; i++) {
			float x = (float)rng.uniform(-1., 1.) * a;
			float y = (float)rng.uniform(-1., 1.) * b;
			points[i].x = cvRound(x * cos_a - y * sin_a + img.cols / 2);
			points[i].y = cvRound(x * sin_a + y * cos_a + img.rows / 2);
		}

		// generate outlier points
		for (; i < count; i++) {
			points[i].x = rng.uniform(0, img.cols);
			points[i].y = rng.uniform(0, img.rows);
		}

		// find the optimal line
		cv::fitLine(points, line, cv::DIST_L1, 1, 0.001, 0.001);

		// draw the points
		img = cv::Scalar::all(0);
		for (i = 0; i < count; i++)
			cv::circle(img, points[i], 2,
				i < count - outliers ? cv::Scalar(0, 0, 255)
				: cv::Scalar(0, 255, 255),
				cv::FILLED, CV_AA, 0);

		// ... and the long enough line to cross the whole image
		d = sqrt((double)line[0] * line[0] + (double)line[1] * line[1]);
		line[0] /= d;
		line[1] /= d;
		t = (float)(img.cols + img.rows);
		pt1.x = cvRound(line[2] - line[0] * t);
		pt1.y = cvRound(line[3] - line[1] * t);
		pt2.x = cvRound(line[2] + line[0] * t);
		pt2.y = cvRound(line[3] + line[1] * t);
		cv::line(img, pt1, pt2, cv::Scalar(0, 255, 0), 3, CV_AA, 0);
		cv::imshow("Fit Line", img);
		key = (char)cv::waitKey(0);
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

	return 0;
}