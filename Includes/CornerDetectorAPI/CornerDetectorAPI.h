#pragma once
#ifndef CORNERDETECTORAPI_H
#define CORNERDETECTORAPI_H

// CornerDetectorAPI.h
// DLL Definition
#include <opencv2/opencv.hpp>
#include "utility.h"
#ifdef CORNERDETECTORAPI_EXPORTS
#define CORNERDETECTOR_API __declspec(dllexport)
#else
#define CORNERDETECTOR_API __declspec(dllimport)
#endif
using namespace cv;
namespace lpdr
{  
  class CornerDetectorFuncs
  {
  public:
    // corner detector
    static CORNERDETECTOR_API  cv::Point findCornerAPI(cv::Mat roi_Gray, int roiIdx, bool hs_leftright, bool vs_topbottom, cornerParameters * conP);        
  };
}


#endif