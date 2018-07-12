

#include "histogramvertical.h"

using namespace cv;
using namespace std;

namespace lpdr
{
  HistogramVertical::HistogramVertical(Mat inputImage, Mat mask)
  {
    analyzeImage(inputImage, mask, true);
  }
}