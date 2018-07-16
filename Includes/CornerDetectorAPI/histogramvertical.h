
#ifndef LPDR_VERTICALHISTOGRAM_H
#define LPPR_VERTICALHISTOGRAM_H

#include "opencv2/imgproc/imgproc.hpp"
#include "histogram.h"
//#include "utility.h"

namespace lpdr
{
  class HistogramVertical : public Histogram
  {
  public:
    HistogramVertical(cv::Mat inputImage, cv::Mat mask);
  };
}
  
#endif // LPDR_VERTICALHISTOGRAM_H // by sangkny
