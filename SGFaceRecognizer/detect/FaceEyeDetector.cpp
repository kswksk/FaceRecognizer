//
//  FaceEyeDetector.cpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 10..
//  Copyright © 2018년 50017516. All rights reserved.
//

#include "FaceEyeDetector.hpp"

FaceEyeDetector::FaceEyeDetector(const string executePath, double scaleFactor, int minNeighbors):
        _executePath(executePath), _scaleFactor(scaleFactor), _minNeighbors(minNeighbors) {
    string algorithm[] = {
        "resources/cascades/haarcascade_eye.xml"
        ,"resources/cascades/haarcascade_mcs_lefteye.xml"
        ,"resources/cascades/haarcascade_mcs_righteye.xml"
        ,"resources/cascades/haarcascade_eye_tree_eyeglasses.xml"
        ,"resources/cascades/haarcascade_lefteye_2splits.xml"
    };
            
    for (int i = 0; i < sizeof(algorithm)/sizeof(string); i++) {
        string strPathEye = format("%s/%s", _executePath.c_str(), algorithm[i].c_str());
        CascadeClassifier c;
        c.load(strPathEye);
        _cascades.push_back(c);
    }
}

FaceEyeDetector::~FaceEyeDetector() {}

void FaceEyeDetector::findEyeInFace(const Mat &img, vector<cv::Rect> &res) {
    Mat tmp;
    cv::Size minScaleSize = cv::Size(20, 20);
    
    //convert the image to grayscale and normalize histogram:
    cvtColor(img, tmp, CV_BGR2GRAY);
    equalizeHist(tmp, tmp);
    
    //clear the vector:
    res.clear();
    
    //detect eye:
    // 눈 위치 찾기
    bool isLeft = false;
    bool isRight = false;
    cv::Rect left, right;
    
    CvRect Quadrant1 = CvRect(img.size().width/2, img.size().height*0.16, img.size().width/2, img.size().height*0.4);
    CvRect Quadrant2 = CvRect(0, img.size().height*0.16, img.size().width/2, img.size().height*0.4);
    
    /*
    rectangle(tmp, cv::Point(Quadrant1.x, Quadrant1.y), cv::Point(Quadrant1.x+Quadrant1.width, Quadrant1.y+Quadrant1.height), Scalar(255,0,0));
    rectangle(tmp, cv::Point(Quadrant2.x, Quadrant2.y), cv::Point(Quadrant2.x+Quadrant2.width, Quadrant2.y+Quadrant2.height), Scalar(0,255,0));
    imshow("test", tmp);
    cvWaitKey();
     */
    
    for (vector<CascadeClassifier>::const_iterator cas = _cascades.begin(); cas != _cascades.end(); cas++) {
        CascadeClassifier cc = *cas;
        vector<cv::Rect> eyes;
        cv::Size maxScaleSize = cv::Size(img.size().width*0.25, img.size().height*0.25);
        cc.detectMultiScale(tmp, eyes, _scaleFactor, _minNeighbors, 0, minScaleSize, maxScaleSize);
        
        for (vector<cv::Rect>::iterator rec = eyes.begin(); rec != eyes.end(); rec++) {
            cv::Rect r = *rec;
            cv::Point p1 = cv::Point(r.x, r.y);
            cv::Point p2 = cv::Point(r.x+r.width, r.y+r.height);
            if (!isRight && isPointInRect(p1, Quadrant1) && isPointInRect(p2, Quadrant1)) {
                right = r;
                isRight = true;
            } else if (!isLeft && isPointInRect(p1, Quadrant2) && isPointInRect(p2, Quadrant2)) {
                left = r;
                isLeft = true;
            }
        }
        
        if (isRight && isLeft) {
            break;
        }
    }
    
    if (isRight && isLeft) {
        res.push_back(left);
        res.push_back(right);
    }
}

bool FaceEyeDetector:: isPointInRect(const cv::Point pt, const cv::Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;
    
    return false;
}
