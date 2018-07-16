
#include "utility.h"
#include <opencv2/core/core.hpp>
#include <functional>
#include <cctype>



using namespace cv;
using namespace std;
/* // parameter descriptions
struct cornerParameters {
  int min_distance_from_corner = 10;				// minimum distance from the corner point
  int max_distance_from_corner = 250; //100			// maximum distance from the cornper point, it affects the effectiveness of sheet's rotation, more the better.
  float corner_min_widthheight_ratio = 0.95;
  float corner_max_widthheight_ratio = 2.75;
  float offsetPerc = 0.10;		// offset % point from the center 0.065
  int adpthres_blockSize = 19; //25	// block size will determine the quality of the corner detection. + -> more edge but more robust edge but lost smooth changing edges
  double adpthres_C = 20;		// decreasing the value results in noisy and more detailed edge detection.
  bool disqualified = false;	// disqualified flag
  std::string reason = "";		// reason
  bool debugTiming = true;
  bool debugGeneral = true;		// debugging options belows
  bool debugGeneralDetails = false; // General debugging Infor in Detail
  bool debugShowImages = true;       // show image files
  bool debugShowCornerImages = true; // shows corner image or not
  bool debugShowImagesDetail = false; // can analyze the detail verson of code with images
};
*/
namespace lpdr
{
  Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY)
  {
    Rect expandedRegion = Rect(original);

    float halfX = round((float) expandXPixels / 2.0);
    float halfY = round((float) expandYPixels / 2.0);
    expandedRegion.x = expandedRegion.x - halfX;
    expandedRegion.width =  expandedRegion.width + expandXPixels;
    expandedRegion.y = expandedRegion.y - halfY;
    expandedRegion.height =  expandedRegion.height + expandYPixels;

	expandedRegion.x = std::min(std::max(expandedRegion.x, 0), maxX);
	
	expandedRegion.y = std::min(std::max(expandedRegion.y, 0), maxY);
	if (expandedRegion.x + expandedRegion.width > maxX)
      expandedRegion.width = maxX - expandedRegion.x;
    if (expandedRegion.y + expandedRegion.height > maxY)
      expandedRegion.height = maxY - expandedRegion.y;
    
    return expandedRegion;
  }
  bool ExpandRect(Rect in, Rect &out, float perScale, Mat src) {
	  /* this function expand given in rect with perScale whin the src image size*/
	  float fs = perScale; // percentage 10/100		
	  // check the validity of the given rectangle which is in the source image	  	  
	  in.x = std::max(0, std::min(in.x, src.cols));
	  in.y = std::max(0, std::min(in.y, src.rows));
	  if (in.x + in.width > src.cols)
		  in.width = src.cols - in.x;
	  if (in.y + in.height > src.rows)
		  in.height = src.rows - in.y;

	  Rect rc(in);
	  out = rc;
	  cv::Size deltaSize(out.width * fs, out.height * fs); // 0.1f = 10/100
	  cv::Point offset(deltaSize.width / 2, deltaSize.height / 2);

	  out += deltaSize;
	  out -= offset;

	  bool is_inside = (out & cv::Rect(0, 0, src.cols, src.rows)) == out;
	  if (is_inside) {
		  return true;
	  }
	  out = rc; // memory leakage checking required !!
	  return is_inside;
  }
  //bool ExpandRect(PlateRegion in, PlateRegion &out, float perScale, Mat src) {	// it is an overloading function of ExpandRect
		//																		/* this function expand given in PlateRegion which has rect and type information using percentage Scale whin the src image size*/
	 // return ExpandRect(in.rect, out.rect, perScale, src);
  //}

  Mat drawImageDashboard(vector<Mat> images, int imageType, unsigned int numColumns)
  {
    unsigned int numRows = ceil((float) images.size() / (float) numColumns);

    Mat dashboard(Size(images[0].cols * numColumns, images[0].rows * numRows), imageType);

    for (unsigned int i = 0; i < numColumns * numRows; i++)
    {
	  if (i < images.size()) {
		 //bug fix by sangkny
		  resize(images[i], images[i], Size(images[0].size()));
		images[i].copyTo(dashboard(Rect((i%numColumns) * images[i].cols, floor((float)i / numColumns) * images[i].rows, images[i].cols, images[i].rows)));
	  }
	  else
      {
        Mat black = Mat::zeros(images[0].size(), imageType);
        black.copyTo(dashboard(Rect((i%numColumns) * images[0].cols, floor((float) i/numColumns) * images[0].rows, images[0].cols, images[0].rows)));
      }
    }

    return dashboard;
  }

  Mat addLabel(Mat input, string label)
  {
    const int border_size = 1;
    const Scalar border_color(0,0,255);
    const int extraHeight = 20;
    const Scalar bg(222,222,222);
    const Scalar fg(0,0,0);

    Rect destinationRect(border_size, extraHeight, input.cols, input.rows);
    Mat newImage(Size(input.cols + (border_size), input.rows + extraHeight + (border_size )), input.type());
    input.copyTo(newImage(destinationRect));

    cout << " Adding label " << label << endl;
    if (input.type() == CV_8U)
      cvtColor(newImage, newImage, CV_GRAY2BGR);

    rectangle(newImage, Point(0,0), Point(input.cols, extraHeight), bg, CV_FILLED);
    putText(newImage, label, Point(5, extraHeight - 5), CV_FONT_HERSHEY_PLAIN  , 0.7, fg);

    rectangle(newImage, Point(0,0), Point(newImage.cols - 1, newImage.rows -1), border_color, border_size);

    return newImage;
  }

  void drawAndWait(cv::Mat frame)
  {
    drawAndWait(&frame);
  }
  void drawAndWait(cv::Mat* frame)
  {
    cv::imshow("Temp Window", *frame);

    while (cv::waitKey(50) == -1)
    {
      // loop
    }

    //cv::destroyWindow("Temp Window");
	//waitKey(0);
	waitKey(1);//
  }

  void displayImage(string windowName, cv::Mat frame)
  {
    
      imshow(windowName, frame);
	  //cv::waitKey(5);
	  waitKey(5);
    
  }

  vector<Mat> produceThresholds(const Mat img_gray/*, Config* config*/)
  {
    const int THRESHOLD_COUNT = 3;
    //Mat img_equalized = equalizeBrightness(img_gray);

    /*timespec startTime;
    getTimeMonotonic(&startTime);*/

    vector<Mat> thresholds;

    for (int i = 0; i < THRESHOLD_COUNT; i++)
      thresholds.push_back(Mat(img_gray.size(), CV_8U));

    int i = 0;

    // Adaptive
    //adaptiveThreshold(img_gray, thresholds[i++], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV , 7, 3);
    //adaptiveThreshold(img_gray, thresholds[i++], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV , 13, 3);
    //adaptiveThreshold(img_gray, thresholds[i++], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV , 17, 3);

    // Wolf
    int k = 0, win=18;
    //NiblackSauvolaWolfJolion (img_gray, thresholds[i++], WOLFJOLION, win, win, 0.05 + (k * 0.35));
    //bitwise_not(thresholds[i-1], thresholds[i-1]);
    NiblackSauvolaWolfJolion (img_gray, thresholds[i++], WOLFJOLION, win, win, 0.05 + (k * 0.35));
    bitwise_not(thresholds[i-1], thresholds[i-1]);

    k = 1;
    win = 22;
    NiblackSauvolaWolfJolion (img_gray, thresholds[i++], WOLFJOLION, win, win, 0.05 + (k * 0.35));
    bitwise_not(thresholds[i-1], thresholds[i-1]);
    //NiblackSauvolaWolfJolion (img_gray, thresholds[i++], WOLFJOLION, win, win, 0.05 + (k * 0.35));
    //bitwise_not(thresholds[i-1], thresholds[i-1]);

    // Sauvola
    k = 1;
    NiblackSauvolaWolfJolion (img_gray, thresholds[i++], SAUVOLA, 12, 12, 0.18 * k);
    bitwise_not(thresholds[i-1], thresholds[i-1]);
    //k=2;
    //NiblackSauvolaWolfJolion (img_gray, thresholds[i++], SAUVOLA, 12, 12, 0.18 * k);
    //bitwise_not(thresholds[i-1], thresholds[i-1]);

   /* if (config->debugTiming)
    {
      timespec endTime;
      getTimeMonotonic(&endTime);
      cout << "  -- Produce Threshold Time: " << diffclock(startTime, endTime) << "ms." << endl;
    }*/

    return thresholds;
    //threshold(img_equalized, img_threshold, 100, 255, THRESH_BINARY);
  }

  double median(int array[], int arraySize)
  {
    if (arraySize == 0)
    {
      //std::cerr << "Median calculation requested on empty array" << endl;
      return 0;
    }

    std::sort(&array[0], &array[arraySize]);
    return arraySize % 2 ? array[arraySize / 2] : (array[arraySize / 2 - 1] + array[arraySize / 2]) / 2;
  }

  Mat equalizeBrightness(Mat img)
  {
    // Divide the image by its morphologically closed counterpart
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(19,19));
    Mat closed;
    morphologyEx(img, closed, MORPH_CLOSE, kernel);

    img.convertTo(img, CV_32FC1); // divide requires floating-point
    divide(img, closed, img, 1, CV_32FC1);
    normalize(img, img, 0, 255, NORM_MINMAX);
    img.convertTo(img, CV_8U); // convert back to unsigned int

    return img;
  }

  void drawRotatedRect(Mat* img, RotatedRect rect, Scalar color, int thickness)
  {
    Point2f rect_points[4];
    rect.points( rect_points );
    for( int j = 0; j < 4; j++ )
      line( *img, rect_points[j], rect_points[(j+1)%4], color, thickness, 8 );
  }

  void fillMask(Mat img, const Mat mask, Scalar color)
  {
    for (int row = 0; row < img.rows; row++)
    {
      for (int col = 0; col < img.cols; col++)
      {
        int m = (int) mask.at<uchar>(row, col);

        if (m)
        {
          for (int z = 0; z < 3; z++)
          {
            int prevVal = img.at<Vec3b>(row, col)[z];
            img.at<Vec3b>(row, col)[z] = ((int) color[z]) | prevVal;
          }
        }
      }
    }
  }

  void drawX(Mat img, Rect rect, Scalar color, int thickness)
  {
    Point tl(rect.x, rect.y);
    Point tr(rect.x + rect.width, rect.y);
    Point bl(rect.x, rect.y + rect.height);
    Point br(rect.x + rect.width, rect.y + rect.height);

    line(img, tl, br, color, thickness);
    line(img, bl, tr, color, thickness);
  }

  double distanceBetweenPoints(Point p1, Point p2)
  {
    float asquared = (p2.x - p1.x)*(p2.x - p1.x);
    float bsquared = (p2.y - p1.y)*(p2.y - p1.y);

    return sqrt(asquared + bsquared);
  }

  float angleBetweenPoints(Point p1, Point p2)
  {
    int deltaY = p2.y - p1.y;
    int deltaX = p2.x - p1.x;

    return atan2((float) deltaY, (float) deltaX) * (180 / CV_PI);
  }

  Size getSizeMaintainingAspect(Mat inputImg, int maxWidth, int maxHeight)
  {
    float aspect = ((float) inputImg.cols) / ((float) inputImg.rows);

    if (maxWidth / aspect > maxHeight)
    {
      return Size(maxHeight * aspect, maxHeight);
    }
    else
    {
      return Size(maxWidth, maxWidth / aspect);
    }
  }


// Compares two strings and computes the edit distance between them
// http://en.wikipedia.org/wiki/Levenshtein_distance
// max is the cutoff (i.e., max distance) where we stop trying to find distance
int levenshteinDistance (const std::string &s1, const std::string &s2, int max)
{
    const char* word1 = s1.c_str();
    int len1 = s1.length();
    const char* word2 = s2.c_str();
    int len2 = s2.length();
    max--;
  
    //int matrix[2][len2 + 1];
    std::vector<std::vector<int> > matrix;
    for (unsigned int i = 0; i < 2; i++)
    {
      std::vector<int> top_elem;
      matrix.push_back(top_elem);
      for (unsigned int j = 0; j < len2 + 1; j++)
        matrix[i].push_back(0);
    }
    int i;
    int j;
    
    /*
      Initialize the 0 row of "matrix".

        0  
        1  
        2  
        3  

     */

    for (j = 0; j <= len2; j++) {
        matrix[0][j] = j;
    }

    /* Loop over column. */
    for (i = 1; i <= len1; i++) {
        char c1;
        /* The first value to consider of the ith column. */
        int min_j;
        /* The last value to consider of the ith column. */
        int max_j;
        /* The smallest value of the matrix in the ith column. */
        int col_min;
        /* The next column of the matrix to fill in. */
        int next;
        /* The previously-filled-in column of the matrix. */
        int prev;

        c1 = word1[i-1];
        min_j = 1;
        if (i > max) {
            min_j = i - max;
        }
        max_j = len2;
        if (len2 > max + i) {
            max_j = max + i;
        }
        col_min = INT_MAX;
        next = i % 2;
        if (next == 1) {
            prev = 0;
        }
        else {
            prev = 1;
        }
        matrix[next][0] = i;
        /* Loop over rows. */
        for (j = 1; j <= len2; j++) {
            if (j < min_j || j > max_j) {
                /* Put a large value in there. */
                matrix[next][j] = max + 1;
            }
            else {
                char c2;

                c2 = word2[j-1];
                if (c1 == c2) {
                    /* The character at position i in word1 is the same as
                       the character at position j in word2. */
                    matrix[next][j] = matrix[prev][j-1];
                }
                else {
                    /* The character at position i in word1 is not the
                       same as the character at position j in word2, so
                       work out what the minimum cost for getting to cell
                       i, j is. */
                    int del;
                    int insert;
                    int substitute;
                    int minimum;

                    del = matrix[prev][j] + 1;
                    insert = matrix[next][j-1] + 1;
                    substitute = matrix[prev][j-1] + 1;
                    minimum = del;
                    if (insert < minimum) {
                        minimum = insert;
                    }
                    if (substitute < minimum) {
                        minimum = substitute;
                    }
                    matrix[next][j] = minimum;
                }
            }
            /* Find the minimum value in the ith column. */
            if (matrix[next][j] < col_min) {
                col_min = matrix[next][j];
            }
        }
        if (col_min > max) {
            /* All the elements of the ith column are greater than the
               maximum, so no match less than or equal to max can be
               found by looking at succeeding columns. */
            return max + 1;
        }
    }
    int returnval = matrix[len1 % 2][len2];
    if (returnval > max + 1)
      returnval = max + 1;
    return returnval;
}
  
  
  LineSegment::LineSegment()
  {
    init(0, 0, 0, 0);
  }

  LineSegment::LineSegment(Point p1, Point p2)
  {
    init(p1.x, p1.y, p2.x, p2.y);
  }

  LineSegment::LineSegment(int x1, int y1, int x2, int y2)
  {
    init(x1, y1, x2, y2);
  }

  void LineSegment::init(int x1, int y1, int x2, int y2)
  {
    this->p1 = Point(x1, y1);
    this->p2 = Point(x2, y2);

    if (p2.x - p1.x == 0)
      this->slope = 0.00000000001;
    else
      this->slope = (float) (p2.y - p1.y) / (float) (p2.x - p1.x);

    this->length = distanceBetweenPoints(p1, p2);

    this->angle = angleBetweenPoints(p1, p2);
  }

  bool LineSegment::isPointBelowLine( Point tp )
  {
    return ((p2.x - p1.x)*(tp.y - p1.y) - (p2.y - p1.y)*(tp.x - p1.x)) > 0;
  }

  float LineSegment::getPointAt(float x)
  {
    return slope * (x - p2.x) + p2.y;
  }

  float LineSegment::getXPointAt(float y)
  {
    float y_intercept = getPointAt(0);
    return (y - y_intercept) / slope;
  }
  
  Point LineSegment::closestPointOnSegmentTo(Point p)
  { // inner product (dot product)
    float top = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y)*(p2.y - p1.y);

    float bottom = distanceBetweenPoints(p2, p1);
    bottom = bottom * bottom;

    float u = top / bottom;

    float x = p1.x + u * (p2.x - p1.x);
    float y = p1.y + u * (p2.y - p1.y);

    return Point(x, y);
  }

  Point LineSegment::intersection(LineSegment line)
  {
    float c1, c2;
    float intersection_X = -1, intersection_Y= -1;

    c1 = p1.y - slope * p1.x; // which is same as y2 - slope * x2

    c2 = line.p2.y - line.slope * line.p2.x; // which is same as y2 - slope * x2

    if( (slope - line.slope) == 0)
    {
      //std::cout << "No Intersection between the lines" << endl;
    }
    else if (p1.x == p2.x)
    {
      // Line1 is vertical
      return Point(p1.x, line.getPointAt(p1.x));
    }
    else if (line.p1.x == line.p2.x)
    {
      // Line2 is vertical
      return Point(line.p1.x, getPointAt(line.p1.x));
    }
    else
    {
      intersection_X = (c2 - c1) / (slope - line.slope);
      intersection_Y = slope * intersection_X + c1;
    }

    return Point(intersection_X, intersection_Y);
  }

  Point LineSegment::midpoint()
  {
    // Handle the case where the line is vertical
    if (p1.x == p2.x)
    {
      float ydiff = p2.y-p1.y;
      float y = p1.y + (ydiff/2);
      return Point(p1.x, y);
    }
    float diff = p2.x - p1.x;
    float midX = ((float) p1.x) + (diff / 2);
    int midY = getPointAt(midX);

    return Point(midX, midY);
  }

  LineSegment LineSegment::getParallelLine(float distance)
  {
    float diff_x = p2.x - p1.x;
    float diff_y = p2.y - p1.y;
    float angle = atan2( diff_x, diff_y);
    float dist_x = distance * cos(angle);
    float dist_y = -distance * sin(angle);

    int offsetX = (int)round(dist_x);
    int offsetY = (int)round(dist_y);

    LineSegment result(p1.x + offsetX, p1.y + offsetY,
                       p2.x + offsetX, p2.y + offsetY);

    return result;
  }
  
  cv::Point findClosestPoint(cv::Point2f* polygon_points, int num_points, cv::Point position)
  {
    int closest_point_index = 0;
    unsigned int smallest_distance = INT_MAX;
    for (unsigned int i = 0; i < num_points; i++)
    {
      Point pos((int)polygon_points[i].x, (int)polygon_points[i].y);
      unsigned int distance = distanceBetweenPoints(pos, position);
      //std::cout << "polys Distance between: " << position << " and " << pos << " = " << distance << endl;
      if (distance < smallest_distance)
      {
        smallest_distance = distance;
        closest_point_index = i;
      }
    }
    
    return Point((int)polygon_points[closest_point_index].x, (int)polygon_points[closest_point_index].y);
  }
  
  std::vector<cv::Point> sortPolygonPoints(cv::Point2f* polygon_points, cv::Size surrounding_image)
  {
    
    vector<Point> return_points;
    
    // Find top-left
    return_points.push_back( findClosestPoint(polygon_points, 4, Point(0, 0)) );
    return_points.push_back( findClosestPoint(polygon_points, 4,Point(surrounding_image.width, 0)) );
    return_points.push_back( findClosestPoint(polygon_points, 4,Point(surrounding_image.width, surrounding_image.height)) );
    return_points.push_back( findClosestPoint(polygon_points, 4,Point(0, surrounding_image.height)) );

    return return_points;
  }
  // Given a contour and a mask, this function determines what percentage of the contour (area)
  // is inside the masked area. 
  float getContourAreaPercentInsideMask(cv::Mat mask, std::vector<std::vector<cv::Point> > contours, std::vector<cv::Vec4i> hierarchy, int contourIndex)
  {
    Mat innerArea = Mat::zeros(mask.size(), CV_8U);
        drawContours(innerArea, contours,
                 contourIndex, // draw this contour
                 cv::Scalar(255,255,255), // in
                 CV_FILLED,
                 8,
                 hierarchy,
                 2
                );
    int startingPixels = cv::countNonZero(innerArea);
    //drawAndWait(&innerArea);

    bitwise_and(innerArea, mask, innerArea);

    int endingPixels = cv::countNonZero(innerArea);
    //drawAndWait(&innerArea);

    return ((float) endingPixels) / ((float) startingPixels);
  }

  std::string toString(int value)
  {
    stringstream ss;
    ss << value;
    return ss.str();
  }
  std::string toString(long value)
  {
    stringstream ss;
    ss << value;
    return ss.str();
  }
  std::string toString(unsigned int value)
  {
    return toString((int) value);
  }
  std::string toString(float value)
  {
    stringstream ss;
    ss << value;
    return ss.str();
  }
  std::string toString(double value)
  {
    stringstream ss;
    ss << value;
    return ss.str();
  }

  std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
  }

// trim from start
  std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
  }

// trim from end
  std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
  }

// trim from both ends
  std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
  }  


// Line Fitting and Corner Detection Function Definition
// linefitting and corner detector
  cv::Point findCorner(cv::Mat roi_Gray, int roiIdx, bool hs_leftright, bool vs_topbottom, cornerParameters * conP) {
    /*vector<Mat> thresholds = produceThresholds(roi_Gray);
    for (int i = 0; i < thresholds.size(); i++) {
    displayImage("threshold : " + to_string(i), thresholds[i]);
    waitKey(1);
    }*/
    assert(roi_Gray.type() == CV_8UC1); // one channel unsigned char
    bool unpaidFull = true;
    Mat binary, temp;
    LineSegment lineHor, lineVer;       // horizontal and vertical lines
    Point FinalCornerPoint(0, 0);        // a final corner point
    int offsetFromCenter = max(10, min(30, (int)((float)roi_Gray.cols * conP->offsetPerc)));
    int adaptiveThreshold_blockSize = conP->adpthres_blockSize;
    double adaptiveThreshold_C = conP->adpthres_C;
    /*
    // block 싸이즈를 키우면 강한 에지만을 잡을 수 있지만 놓치는 에지가 생길 수 있다. 하지만, 잡음은 없앨 수 있다.
    // threshold C (mean에서 빼주는 값)를 높이면 threshold 가 낮아지는 효과가 나기 때문에 잡음이 많고 세세한 부분까지 잡을 수 있음.
    // 마지막 term인 C를 많이 낮추면 opening을 해야 하는 수가 생김..
    // 300x300 의 경우는 25, 30 이 좋은 효과가 있음.
    // 700x700의 경우는 35, 또는 C를 20으로 주면 대부분의 에지를 잡음.
    */

    if (unpaidFull && (conP->debugGeneralDetails || conP->debugShowCornerImages == false || conP->debugShowImages == false || conP->debugShowImagesDetail)) {
      cout << " ------>>  Coner Detector Version 1.03 DLL version <<-------" << endl;
      cout << "The current version only supports the limited debugging information.!!" << endl;
      cout << "For more information, Please Contact Author (sangkny@gmail.com) to BUY a full COPYRIGHT." << endl;
      std::getchar();
      return FinalCornerPoint;
    }

    //adaptiveThreshold(roi_Gray, binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 30); 300x300
    adaptiveThreshold(roi_Gray, binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, adaptiveThreshold_blockSize/*25*/, adaptiveThreshold_C/*20*/);
    if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
      lpdr::displayImage("crop_gray " + to_string(roiIdx), roi_Gray);
      lpdr::displayImage("Binary " + to_string(roiIdx), binary);
    }

    Mat histMask = Mat::ones(binary.size(), binary.type());
    HistogramVertical vhist(binary, histMask);
    HistogramHorizontal hhist(binary, histMask);
    if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
      lpdr::displayImage("Ver histogram" + to_string(roiIdx), vhist.histoImg);
      lpdr::displayImage("Hor histogram" + to_string(roiIdx), hhist.histoImg);
    }
    int maxIdxx = vhist.getLocalMaximum(1, binary.cols - 1); // histogram maximum bin index which indicates the location of maximumu histogram peak.
    int maxIdxy = hhist.getLocalMaximum(1, binary.rows - 1);
    int maxValuex = vhist.getHeightAt(maxIdxx);
    int maxValuey = hhist.getHeightAt(maxIdxy);
    // make cases according to the hs_/vs_ conditions (4 cases)  
    int startX, endX, dX, startY, endY, dY;
    int line_sX, line_endX, line_sY, line_endY;

    /* Top-Left Region Search */
    if (hs_leftright == false && vs_topbottom == false)
    {
      // top-left search
      int xEdge = -1;
      int yRef = binary.rows - offsetFromCenter; // y fixed
      int peak_loc_variancex = 5; // 			
      for (int xi = binary.cols - 1; xi >= 0; xi--) {
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          line(tempImg, Point(0, yRef), Point(binary.cols - 1, yRef), Scalar(0, 255, 255), 1, 8); // yellow
          line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);		// red line
          imshow("Detected line (x,y) > 0 for vertical line searching", tempImg);
          //cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
          cvWaitKey(1);
        }
        if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi)>0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi)>0) {
          xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
          break;
        }
      }
      if (xEdge == -1) {
        cout << "Edge is not found. Please check the ROIin the Top-Left Region!!" << endl;
      }
      else {
        // search the histogram peak and its length in Binary image or make a line      
        if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
          lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // 
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
          int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
          int maxValue1 = vhist.getHeightAt(maxIdx1);
          lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
          if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
      }
      // top-left field search	to finde a horizontal line			
      int yEdge = -1;
      int xRef;
      xRef = (xEdge == -1) ? binary.cols - offsetFromCenter : ((xEdge + (int)(float)((binary.cols - xEdge) + 1) / 2.)); // x fixed, from bottom to up direction
      int peak_loc_variancey = 5; // 			
      for (int yi = binary.rows - 1; yi >= 0; yi--) {
        //for (int yi = 0;yi< binary.rows - 1; yi++) { // sangkny dif
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          //tempImg = Mat::zeros(binary.size(), binary.type());
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
          line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
          line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 for horizontal line in the top-left region", tempImg);
          //cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")" "-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
          cvWaitKey(1);
        }
        if ((int)binary.at<uchar>(yi, xRef) > 0 || /*(int)binary.at<uchar>(yi, max(0, min(binary.cols -1, xRef- offsetFromCenter))) > 0 ||*/ (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0) {
          yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
          break;
        }
      }
      if (yEdge == -1) {
        cout << "Edge is not found to y direction. Please check the ROI in the Top-Left Region !!" << endl;
      }
      else {
        // search the histogram peak and its length in Binary image or make a line      
        if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// vertical line in a range
          lineHor = LineSegment(Point(binary.cols - maxValuey, maxIdxy), Point(xRef, maxIdxy));
          if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
               // separate this region into two regions
          int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
          int maxValue1 = hhist.getHeightAt(maxIdx1);
          lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
          if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
          }
        }
      }

    }

    /* Top-Right region (it has left and right side) */
    if (hs_leftright == true && vs_topbottom == false)
    {
      // top-right field search
      int xEdge = -1;
      int yRef = binary.rows - offsetFromCenter; // y fixed
      int peak_loc_variancex = 5; // 			
      for (int xi = 0; xi< binary.cols; xi++) {        // dif for (int xi = binary.cols - 1; xi >= 0; xi--) {
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          line(tempImg, Point(0, yRef), Point(binary.cols - 1, yRef), Scalar(0, 255, 255), 1, 8);
          line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
          //cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
          cvWaitKey(1);
        }
        if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
          xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
          break;
        }
      }
      if (xEdge == -1) {
        cout << "Edge is not found. Please check the ROI in Top-Right Region!!" << endl;
      }
      else {
        // search the histogram peak and its length in Binary image or make a line      
        if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
          lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // no changes line from top to bottom direction
          if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
          int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
          int maxValue1 = vhist.getHeightAt(maxIdx1);
          lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
          if (conP->debugShowImagesDetail&& conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
      }
      // top-right and right side field search to detect a horizontal line
      int yEdge = -1;
      int xRef;
      xRef = (xEdge == -1) ? offsetFromCenter : min(offsetFromCenter, (xEdge + 1) / 2); // //sangkny dif x fixed, from bottom to up direction, left side was binary.cols - 10;
      int peak_loc_variancey = 5; // 			
      for (int yi = binary.rows - 1; yi >= 0; yi--) {
        //for (int yi = 0;yi< binary.rows - 1; yi++) {
        if (conP->debugShowImagesDetail && conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          //tempImg = Mat::zeros(binary.size(), binary.type());
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
          line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
          line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 for horizontal line in the top-right region", tempImg);
          //cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
          cvWaitKey(1);
        }
        if ((int)binary.at<uchar>(yi, xRef) > 0 || (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 /*|| (int)binary.at<uchar>(yi,max(0, min(binary.cols-1,xRef+ offsetFromCenter))) > 0 */) {
          yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
          break;
        }
      }
      if (yEdge == -1) {
        cout << "Edge is not found to y direction. Please check the ROI in Top-Right Regin" << endl;
        // something wrong message
      }
      else {
        // search the histogram peak and its length in Binary image or make a line     
        if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
          lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
               // separate this region into two region
          int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
          int maxValue1 = hhist.getHeightAt(maxIdx1);
          lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
          }
        }
      }
    }

    /* Bottom-Left Region Search */
    if (hs_leftright == false && vs_topbottom == true) // Bottom-Left region
    {	  // bottom-left field search
      int xEdge = -1;
      int yRef = offsetFromCenter;		// y location fixed  // sangkny dif
      int peak_loc_variancex = 5; // 			
      for (int xi = binary.cols - 1; xi >= 0; xi--) { // dif different code region from other cases
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
          //cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
          cvWaitKey(1);
        }
        if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
          xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
          break;
        }
      }
      if (xEdge == -1) {
        cout << "Edge is not found. Please check the ROI! in Bottom-Left Region" << endl;
      }
      else {
        // search the histogram peak and its length in Binary image or make a line		  
        if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
          lineVer = LineSegment(Point(maxIdxx, binary.rows - maxValuex), Point(maxIdxx, yRef)); // no changes line from top to bottom direction
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
          int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
          int maxValue1 = vhist.getHeightAt(maxIdx1);
          lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
      }
      // Detecting a horizontal line in the Bottom-Left Region
      int yEdge = -1;
      int xRef; // = binary.cols - offsetFromCenter; // //sangkny dif x fixed, from top-to-bottom(true) direction, the left side was binary.cols - 10;
      xRef = (xEdge == -1) ? binary.cols - offsetFromCenter : ((xEdge + (int)(float)((binary.cols - xEdge) + 1) / 2.)); // x fixed, from bottom to up direction
      int peak_loc_variancey = 5; // 			
                                  //for (int yi = binary.rows - 1; yi >= 0; yi--) {
      for (int yi = 0; yi< binary.rows - 1; yi++) {			// sangkny dif 
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          //tempImg = Mat::zeros(binary.size(), binary.type());
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);			  
          line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
          line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 in the Bottom-Left Region", tempImg);
          //cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
          cvWaitKey(1);
        }
        if ((int)binary.at<uchar>(yi, xRef) > 0 || /*(int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 ||*/ (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0) {
          yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
          break;
        }
      }
      if (yEdge == -1) {
        cout << "Edge is not found to search a horizontal line in the y direction. Please check the ROI!!" << endl;
        // something wrong message
      }
      else {
        // search the histogram peak and its length in Binary image or make a line
        if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
          lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
               // separate this region into two region
          int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
          int maxValue1 = hhist.getHeightAt(maxIdx1);
          lineHor = LineSegment(Point(binary.rows - maxValue1, maxIdx1), Point(xRef, maxIdx1));
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
          }
        }
      }
    }

    /* Bottom-Right Region Search Algorithm */
    if (hs_leftright == true && vs_topbottom == true) // Bottom-Right Region Definition
    {
      // Bottom-left side,  a vertical line search
      int xEdge = -1;
      int yRef = offsetFromCenter;				// y fixed  : dif
      int peak_loc_variancex = 5;	// 			
      for (int xi = 0; xi< binary.cols; xi++) {        // dif for (int xi = binary.cols - 1; xi >= 0; xi--) {
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          line(tempImg, Point(xi, 0), Point(xi, binary.rows - 1), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 to search a vertical line", tempImg);
          //cout << "Point (x,y) > 0 : (" << yRef << ", " << xi << ")" "-> value:" << to_string(binary.at<uchar>(yRef, xi)) << endl;
          cvWaitKey(1);
        }
        if (binary.at<uchar>(max(0, min(binary.rows - 1, yRef - offsetFromCenter)), xi) > 0 || binary.at<uchar>(yRef, xi) > 0 || binary.at<uchar>(max(0, min(binary.rows - 1, yRef + offsetFromCenter)), xi) > 0) {
          xEdge = xi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary
          break;
        }
      }
      if (xEdge == -1) {
        cout << "Edge is not found to detect a vertical line in the Bottom-Right Region. Please check the ROI !!" << endl;
        // disqualified
      }
      else {
        // search the histogram peak and its length in Binary image or make a line
        if (abs(maxIdxx - xEdge) <= peak_loc_variancex) {	// vertical line search in a range
          lineVer = LineSegment(Point(maxIdxx, yRef), Point(maxIdxx, maxValuex)); // diff no changes line from top to bottom direction
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("1st detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
          int maxIdx1 = vhist.getLocalMaximum(max(0, xEdge - (peak_loc_variancex + 1) / 2), min(binary.cols - 1, xEdge + (peak_loc_variancex + 1) / 2));
          int maxValue1 = vhist.getHeightAt(maxIdx1);
          lineVer = LineSegment(Point(maxIdx1, binary.rows - maxValue1), Point(maxIdx1, yRef));
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineVer.p1, lineVer.p2, Scalar(255, 0, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a vertical direction->" + to_string(roiIdx), debugTempImg);
          }
        }
      }
      // In the Bottom-Right Region, and Right side field search to detect a horizontal line
      int yEdge = -1;
      int xRef; // = offsetFromCenter; // //sangkny dif x fixed, from bottom to up direction, left side was binary.cols - 10;
      xRef = (xEdge == -1) ? offsetFromCenter : min(offsetFromCenter, (xEdge + 1) / 2); // this code is more reasonable because the detected vertical line is robust and important
      int peak_loc_variancey = 5; // 			
                                  //for (int yi = binary.rows - 1; yi >= 0; yi--) {
      for (int yi = 0; yi< binary.rows - 1; yi++) {					// sangkny dif
        if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
          Mat tempImg;
          tempImg = binary.clone();
          //tempImg = Mat::zeros(binary.size(), binary.type());
          if (tempImg.channels() < 3)
            cvtColor(tempImg, tempImg, CV_GRAY2BGR);
          //temp.at<uchar>(yi,xRef) = binary.at<uchar>(yi,xRef);
          line(tempImg, Point(xRef, 0), Point(xRef, binary.rows - 1), Scalar(0, 255, 255), 1, 8);
          line(tempImg, Point(0, yi), Point(binary.cols - 1, yi), Scalar(0, 0, 255), 2, 8);
          imshow("Detected line (x,y) > 0 in the Bottom-Right Region", tempImg);
          //cout << "Point (x,y) > 0 : (" << xRef << ", " << yi << ")-> value:" << to_string(binary.at<uchar>(yi, xRef)) << endl;
          cvWaitKey(1);
        }
        if ((int)binary.at<uchar>(yi, xRef) > 0 || (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef - offsetFromCenter))) > 0 /*|| (int)binary.at<uchar>(yi, max(0, min(binary.cols - 1, xRef + offsetFromCenter))) > 0*/) {
          yEdge = yi; // the first location whihc is not 0,and it will be the end pixel of + battery boundary            
          break;
        }
      }
      if (yEdge == -1) {
        cout << "Edge is not found to detection a horizontal line in the y direction for Bottom-Right Region. Please check the ROI!!. Disqualified" << endl;
        // something wrong message  // disqualified
      }
      else {
        // search the histogram peak and its length in Binary image or make a line    
        if (abs(maxIdxy - yEdge) <= peak_loc_variancey) {	// horizontal line in a range
          lineHor = LineSegment(Point(xRef, maxIdxy), Point(binary.cols - maxValuey, maxIdxy)); // sangkny dif : switch the x locations from inside to out size
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("1st detected line in a horizontal direction => " + to_string(roiIdx), debugTempImg);
          }
        }
        else { // the first peak is not related to the the boundary we are looking for, therefore, we need to find out again
               // separate this region into two region
          int maxIdx1 = hhist.getLocalMaximum(max(0, yEdge - (peak_loc_variancey + 1) / 2), min(binary.rows - 1, yEdge + (peak_loc_variancey + 1) / 2));
          int maxValue1 = hhist.getHeightAt(maxIdx1);
          lineHor = LineSegment(Point(xRef, maxIdx1), Point(binary.rows - maxValue1, maxIdx1));
          if (conP->debugShowImagesDetail&&conP->debugGeneral && conP->debugShowImages) {
            Mat debugTempImg = binary.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            line(debugTempImg, lineHor.p1, lineHor.p2, Scalar(0, 255, 0), 2, 8);
            lpdr::displayImage("2nd detected line in a horizontal direction=> " + to_string(roiIdx), debugTempImg);
          }
        }
      }
    }

    /*---> Corner Point Detection Condition and Detection <--*/
    if (lineHor.length > 0 && lineVer.length > 0) {
      // corner point
      Point corPt = lineHor.intersection(lineVer);
      // verify corner point
      bool bconfirmedCornerPt = verifyCornerPoint(binary, hs_leftright, vs_topbottom, lineHor, lineVer, corPt, conP);

      if (bconfirmedCornerPt && corPt.x > 0 && corPt.x < binary.cols && corPt.y > 0 && corPt.y < binary.rows) { // more condition is required
        FinalCornerPoint = corPt;
        if (conP->debugGeneral) {
          cout << "Corner Point:(x,y) -> (" << FinalCornerPoint.x << ", " << FinalCornerPoint.y << ")." << endl;
          if (conP->debugShowCornerImages) {
            Mat debugTempImg = roi_Gray.clone();
            if (debugTempImg.channels() < 3)
              cvtColor(debugTempImg, debugTempImg, CV_GRAY2BGR);
            circle(debugTempImg, FinalCornerPoint, 3, Scalar(0, 0, 255), 2, 8);
            //displayImage("Corner Points:" + to_string(roiIdx), debugTempImg);
            std::string win_Name = "Corner Points:";
            cv::namedWindow(win_Name + to_string(roiIdx), cv::WINDOW_NORMAL);
            cv::imshow(win_Name + to_string(roiIdx), debugTempImg);
            int showWindowWidth = min(roi_Gray.cols, 300);
            cv::resizeWindow(win_Name + to_string(roiIdx), showWindowWidth, (int)((float)roi_Gray.rows / ((float)roi_Gray.cols / (float)(showWindowWidth))));
            cv::waitKey(1);
          }
        }
      }
      else {
        FinalCornerPoint = Point(0, 0);
        if (conP->debugGeneral) {
          if (roiIdx == 0)
            cout << "Left Region: Corner Vefirication is failed !!" << endl;
          else
            cout << "Right Region: Corner Vefirication is failed !!" << endl;
        }
      }
    }

    return FinalCornerPoint;
  }

  bool verifyCornerPoint(Mat &Binary, bool hs_leftright, bool vs_topbottom, LineSegment &h_line, LineSegment &v_line, Point &cornerPt, cornerParameters *conP) {
    // case study according to the ROI region
    assert(Binary.type() == CV_8UC1); // should be binary (0, 255)
    int h_dist = -1, v_dist = -1;
    int x_dist_Idx = -1, y_dist_Idx = -1;
    int sPtX = cornerPt.x;            // starting point (x,y)
    int sPtY = cornerPt.y;
    int min_distance = conP->min_distance_from_corner;
    int max_distance = conP->max_distance_from_corner;
    float corner_MaxWidthHeight_Ratio = conP->corner_max_widthheight_ratio;
    float corner_MinWidthHeight_Ratio = conP->corner_min_widthheight_ratio;
    if (hs_leftright == false && vs_topbottom == false) {       // Top-Left ROI Region 
                                                                //horizontal and vertical distance computation by turns
                                                                // x-> +, y -> +
      for (int xi = sPtX; xi <= min(Binary.cols - 1, sPtX + max_distance); xi++) {
        if (Binary.at<uchar>(sPtY, xi) > 0) {
          x_dist_Idx = xi;
          break;
        }
      }
      if (x_dist_Idx == -1) {
        if (conP->debugGeneral && conP->debugGeneralDetails)
          cout << "TL-Region, x_dist_Idx is failed. Please increase the max_distance_from_corner parameter to give it more flexibility!!" << endl;
        return false;
      }
      for (int yi = sPtY; yi <= min(Binary.rows - 1, sPtY + max_distance); yi++) {
        if (Binary.at<uchar>(yi, sPtX) > 0) {
          y_dist_Idx = yi;
          break;
        }
      }
      h_dist = abs(x_dist_Idx - sPtX);
      v_dist = abs(y_dist_Idx - sPtY);
      if (conP->debugGeneral && conP->debugGeneralDetails) { // debugging
        cout << "TL-Region:" << endl;
        cout << "(x_dist_Idx, y_dist_Idx, h_dist, v_dist)-- > (" << x_dist_Idx << ", " << y_dist_Idx << ", " << h_dist << ", " << v_dist << ")" << endl;
      }
      if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) /*&& (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist))*/)
        return true;
    }
    else if (hs_leftright == true && vs_topbottom == false) {    // Top-Right ROI Region  
                                                                 // x-> -, y-> +
      for (int xi = sPtX; xi >= max(0, sPtX - max_distance); xi--) {
        if (Binary.at<uchar>(sPtY, xi) > 0) {
          x_dist_Idx = xi;
          break;
        }
      }
      if (x_dist_Idx == -1) {
        if (conP->debugGeneral && conP->debugGeneralDetails)
          cout << "TR-Region, x_dist_Idx is failed. Please increase the max_distance_from_corner parameter to give it more flexibility!!" << endl;
        return false;
      }
      for (int yi = sPtY; yi <= min(Binary.rows - 1, sPtY + max_distance); yi++) {
        if (Binary.at<uchar>(yi, sPtX) > 0) {
          y_dist_Idx = yi;
          break;
        }
      }
      h_dist = abs(x_dist_Idx - sPtX);
      v_dist = abs(y_dist_Idx - sPtY);
      if (conP->debugGeneral && conP->debugGeneralDetails) { // debugging
        cout << "TR-Region:" << endl;
        cout << "(x_dist_Idx, y_dist_Idx, h_dist, v_dist)-- > (" << x_dist_Idx << ", " << y_dist_Idx << ", " << h_dist << ", " << v_dist << ")" << endl;
      }
      if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) /*&& (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist))*/)
        return true;

    }
    else if (hs_leftright == false && vs_topbottom == true) {    // Bottom-Left ROI Region
                                                                 // x-> +, y -> -
      for (int xi = sPtX; xi <= min(Binary.cols - 1, sPtX + max_distance); xi++) {
        if (Binary.at<uchar>(sPtY, xi) > 0) {
          x_dist_Idx = xi;
          break;
        }
      }
      if (x_dist_Idx == -1) {
        if (conP->debugGeneral && conP->debugGeneralDetails)
          cout << "BL-Region, x_dist_Idx is failed. Please increase the max_distance_from_corner parameter to give it more flexibility!!" << endl;
        return false;
      }
      for (int yi = sPtY; yi >= max(0, sPtY - max_distance); yi--) {
        if (Binary.at<uchar>(yi, sPtX) > 0) {
          y_dist_Idx = yi;
          break;
        }
      }
      h_dist = abs(x_dist_Idx - sPtX);
      v_dist = abs(y_dist_Idx - sPtY);
      if (conP->debugGeneral && conP->debugGeneralDetails) { // debugging
        cout << "BL-Region:" << endl;
        cout << "(x_dist_Idx, y_dist_Idx, h_dist, v_dist) -->(" << x_dist_Idx << ", " << y_dist_Idx << ", " << h_dist << ", " << v_dist << ")" << endl;
      }
      if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) /*&& (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist))*/)
        return true;
    }
    else if (hs_leftright == true && vs_topbottom == true) {     // Bottom-Right ROI region
                                                                 // x-> -, y -> -
      for (int xi = sPtX; xi >= max(0, sPtX - max_distance); xi--) {
        if (Binary.at<uchar>(sPtY, xi) > 0) {
          x_dist_Idx = xi;
          break;
        }
      }
      if (x_dist_Idx == -1) {
        if (conP->debugGeneral && conP->debugGeneralDetails)
          cout << "BR-Region, x_dist_Idx is failed. Please increase the max_distance_from_corner parameter to give it more flexibility!!" << endl;
        return false;
      }
      for (int yi = sPtY; yi >= max(0, sPtY - max_distance); yi--) {
        if (Binary.at<uchar>(yi, sPtX) > 0) {
          y_dist_Idx = yi;
          break;
        }
      }
      h_dist = abs(x_dist_Idx - sPtX);
      v_dist = abs(y_dist_Idx - sPtY);
      if (conP->debugGeneral && conP->debugGeneralDetails) { // debugging
        cout << "BR-Region:" << endl;
        cout << "(x_dist_Idx, y_dist_Idx, h_dist, v_dist)-- > (" << x_dist_Idx << ", " << y_dist_Idx << ", " << h_dist << ", " << v_dist << ")" << endl;
      }
      if (y_dist_Idx != -1 && h_dist >= min_distance && v_dist >= min_distance && (h_dist >= (int)(corner_MinWidthHeight_Ratio*(float)v_dist)) /*&& (h_dist < (int)(corner_MaxWidthHeight_Ratio*(float)v_dist))*/)
        return true;

    }
    else {
      assert(false);
    }
    return false;
  }

}