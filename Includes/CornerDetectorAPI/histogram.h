
#ifndef LPDR_HISTOGRAM_H
#define LPDR_HISTOGRAM_H

#include "opencv2/imgproc/imgproc.hpp"
//#include "utility.h"

namespace lpdr
{

  class Histogram
  {
  public:
    Histogram();
    virtual ~Histogram();

    cv::Mat histoImg;

    // Returns the lowest X position between two points.
    int getLocalMinimum(int leftX, int rightX);
    // Returns the highest X position between two points.
    int getLocalMaximum(int leftX, int rightX);

    int getHeightAt(int x);

    std::vector<std::pair<int, int> > get1DHits(int yOffset);

  protected:

    std::vector<int> colHeights;


    void analyzeImage(cv::Mat inputImage, cv::Mat mask, bool use_y_axis);

    int detect_peak(const double *data, int data_count, int *emi_peaks,
                    int *num_emi_peaks, int max_emi_peaks, int *absop_peaks,
                    int *num_absop_peaks, int max_absop_peaks, double delta,
                    int emi_first);
  };

}
#endif //LPDR_HISTOGRAM_H
