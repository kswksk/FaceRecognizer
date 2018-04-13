//
//  FaceEyeDetector.hpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 10..
//  Copyright © 2018년 50017516. All rights reserved.
//

#ifndef FaceEyeDetector_hpp
#define FaceEyeDetector_hpp

#include "CommonHeader.h"

class FaceEyeDetector {
public :
    FaceEyeDetector(const string executePath, double scaleFactor = DET_SCALE_FACTOR,
                    int minNeighbors = DET_MIN_NEIGHBORS);
    virtual ~FaceEyeDetector();
    void findEyeInFace(const Mat &img, vector<cv::Rect> &res);
private :
    vector<CascadeClassifier> _cascades;
    double _scaleFactor;
    int _minNeighbors;
    string _executePath;
    
    bool isPointInRect(const cv::Point pt, const cv::Rect rc);
};

#endif /* FaceEyeDetector_hpp */
