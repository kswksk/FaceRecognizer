//
//  FaceDetect.hpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#ifndef FaceDetector_hpp
#define FaceDetector_hpp

#include "CommonHeader.h"

using namespace std;
using namespace cv;

class FaceDetector {
public :
    FaceDetector(
        const string &cascadePath,
        double scaleFactor = DET_SCALE_FACTOR,
        int minNeighbors = DET_MIN_NEIGHBORS
    );
    virtual ~FaceDetector();
    void findFacesInImage(const Mat &img, vector<cv::Rect> &res);
private :
    CascadeClassifier _cascade;
    double _scaleFactor;
    int _minNeighbors;
};

#endif /* FaceDetector_hpp */
