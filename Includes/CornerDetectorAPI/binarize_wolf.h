
#ifndef LPDR_BINARIZEWOLF_H
#define LPDR_BINARIZEWOLF_H

//#include "support/filesystem.h"

#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <iostream>


namespace lpdr
{

  enum NiblackVersion
  {
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
  };

  #define BINARIZEWOLF_VERSION	"2.3 (February 26th, 2013)"
  #define BINARIZEWOLF_DEFAULTDR 128

  #define uget(x,y)    at<unsigned char>(y,x)
  #define uset(x,y,v)  at<unsigned char>(y,x)=v;
  #define fget(x,y)    at<float>(y,x)
  #define fset(x,y,v)  at<float>(y,x)=v;

  void NiblackSauvolaWolfJolion (cv::Mat im, cv::Mat output, NiblackVersion version,
                                 int winx, int winy, double k, double dR=BINARIZEWOLF_DEFAULTDR);

}

#endif // OPENLPDR_BINARIZEWOLF_H
