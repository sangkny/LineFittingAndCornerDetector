// CornerDetectorAPI.cpp : DLL 응용 프로그램을 위해 내보낸 함수를 정의합니다.
//

#include "stdafx.h"
#include "CornerDetectorAPI.h"

using namespace std;
using namespace lpdr;
using namespace cv;

namespace lpdr
{  
    Point CornerDetectorFuncs::findCornerAPI(Mat roi_Gray, int roiIdx, bool hs_leftright, bool vs_topbottom, cornerParameters * conP) {
      return  findCorner(roi_Gray, roiIdx, hs_leftright, vs_topbottom, conP);
    } 
    /*double CornerDetectorFuncs::Add(double a, double b) {
      return a + b;
    }*/
}


