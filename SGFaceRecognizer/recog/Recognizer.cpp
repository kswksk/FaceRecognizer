//
//  FaceRecognizer.cpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#include "Recognizer.hpp"

Recognizer::Recognizer(const vector<Mat> imgs, const vector<int> labels, int radius, int neighbors,
                       int grid_x, int grid_y, double threshold) {
    // vector<int> labels(imgs.size());
    // for (vector<int>::iterator it = labels.begin(); it != labels.end(); *(it++) = PERSON_LABEL);
    
    _faceSize = cv::Size(imgs[0].size().width, imgs[0].size().height);
    _model = LBPHFaceRecognizer::create(radius, neighbors, grid_x, grid_y, threshold);
    // _model = EigenFaceRecognizer::create();
    // _model = FisherFaceRecognizer::create();
    _model->train(imgs, labels);
}

Recognizer::~Recognizer() {}

int Recognizer::recognize(const Mat &face, double &confidence) const {
    Mat gray;
    int label;
    cvtColor(face, gray, CV_BGR2GRAY);
    resize(gray, gray, _faceSize);
    _model->predict(gray, label, confidence);
    
    /*
    if (confidence > 110) {
        label = -1;
    }
    */
    
    return label;
}
