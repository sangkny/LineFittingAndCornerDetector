

#ifndef OPENALPR_HISTOGRAMHORIZONTAL_H
#define OPENALPR_HISTOGRAMHORIZONTAL_H

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
#endif //OPENALPR_HISTOGRAMHORIZONTAL_H
