// CornerDetectorAPI.cpp : DLL ���� ���α׷��� ���� ������ �Լ��� �����մϴ�.
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


