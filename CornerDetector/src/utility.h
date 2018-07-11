
#ifndef LPDR_UTILITY_H
#define LPDR_UTILITY_H

#include <iostream>
#include <stdio.h>
#include <string.h>

//#include "constants.h"
//#include "support/timing.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "binarize_wolf.h"
#include <vector>
//#include "config.h"
//#include "detector_types.h"

namespace lpdr
{

  class LineSegment
  {

    public:
      cv::Point p1, p2;
      float slope;
      float length;
      float angle;

      // LineSegment(Point point1, Point point2);
      LineSegment();
      LineSegment(int x1, int y1, int x2, int y2);
      LineSegment(cv::Point p1, cv::Point p2);

      void init(int x1, int y1, int x2, int y2);

      bool isPointBelowLine(cv::Point tp);

      float getPointAt(float x);
      float getXPointAt(float y);

      cv::Point closestPointOnSegmentTo(cv::Point p);

      cv::Point intersection(LineSegment line);

      LineSegment getParallelLine(float distance);

      cv::Point midpoint();

      inline std::string str()
      {
        std::stringstream ss;
        ss << "(" << p1.x << ", " << p1.y << ") : (" << p2.x << ", " << p2.y << ")";
        return ss.str() ;
      }

  };

  double median(int array[], int arraySize);

  std::vector<cv::Mat> produceThresholds(const cv::Mat img_gray/*, Config* config*/);

  cv::Mat drawImageDashboard(std::vector<cv::Mat> images, int imageType, unsigned int numColumns);

  void displayImage(std::string windowName, cv::Mat frame);
  void drawAndWait(cv::Mat frame);
  void drawAndWait(cv::Mat* frame);

  double distanceBetweenPoints(cv::Point p1, cv::Point p2);

  void drawRotatedRect(cv::Mat* img, cv::RotatedRect rect, cv::Scalar color, int thickness);

  void drawX(cv::Mat img, cv::Rect rect, cv::Scalar color, int thickness);
  void fillMask(cv::Mat img, const cv::Mat mask, cv::Scalar color);

  float angleBetweenPoints(cv::Point p1, cv::Point p2);

  cv::Size getSizeMaintainingAspect(cv::Mat inputImg, int maxWidth, int maxHeight);

  float getContourAreaPercentInsideMask(cv::Mat mask, std::vector<std::vector<cv::Point> > contours, std::vector<cv::Vec4i> hierarchy, int contourIndex);

  cv::Mat equalizeBrightness(cv::Mat img);

  cv::Rect expandRect(cv::Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY); 
  bool ExpandRect(cv::Rect in, cv::Rect &out, float perScale, cv::Mat src);		/* this function expand given in rect with perScale whin the src image size */
  //bool ExpandRect(PlateRegion in, PlateRegion &out, float perScale, cv::Mat src);		/* this function expand given in rect with perScale whin the src image size */

  cv::Mat addLabel(cv::Mat input, std::string label);

  // Given 4 random points (Point2f array), order them as top-left, top-right, bottom-right, bottom-left
  // Useful for orienting a rotatedrect
  std::vector<cv::Point> sortPolygonPoints(cv::Point2f* polygon_points, cv::Size surrounding_image);
  cv::Point findClosestPoint(cv::Point2f* polygon_points, int num_points, cv::Point position);
  
  int levenshteinDistance (const std::string &s1, const std::string &s2, int max);
  std::string toString(int value);
  std::string toString(long value);
  std::string toString(unsigned int value);
  std::string toString(float value);
  std::string toString(double value);

  std::string replaceAll(std::string str, const std::string& from, const std::string& to);
  
  std::string &ltrim(std::string &s);
  std::string &rtrim(std::string &s);
  std::string &trim(std::string &s);
}

#endif // OPENLPDR_UTILITY_H
