

#ifndef LPDR_HISTOGRAMHORIZONTAL_H
#define LPDR_HISTOGRAMHORIZONTAL_H

#include "opencv2/imgproc/imgproc.hpp"
#include "histogram.h"


namespace lpdr
{
  class HistogramHorizontal : public Histogram
  {
  public:
    HistogramHorizontal(cv::Mat inputImage, cv::Mat mask);
  };

}
#endif //LPDR_HISTOGRAMHORIZONTAL_H
