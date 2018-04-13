//
//  FaceDetect.cpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#include "FaceDetector.hpp"

FaceDetector::FaceDetector(
       const string &cascadePath,
       double scaleFactor,
       int minNeighbors):
    _scaleFactor(scaleFactor), _minNeighbors(minNeighbors)
{
    _cascade.load(cascadePath);
}


FaceDetector::~FaceDetector() {}

void FaceDetector::findFacesInImage(const Mat &img, vector<cv::Rect> &res) {
    Mat tmp;
    cv::Size minScaleSize = cv::Size(40, 40);
    
    //convert the image to grayscale and normalize histogram:
    cvtColor(img, tmp, CV_BGR2GRAY);
    equalizeHist(tmp, tmp);
    
    //clear the vector:
    res.clear();
    
    //detect faces:
    _cascade.detectMultiScale(tmp, res, _scaleFactor, _minNeighbors, 0, minScaleSize);
}
