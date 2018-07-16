

#include "histogramhorizontal.h"

namespace lpdr
{
  HistogramHorizontal::HistogramHorizontal(cv::Mat inputImage, cv::Mat mask) {

      analyzeImage(inputImage, mask, false);
  }
}