//
//  FaceRecognizer.hpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#ifndef Recognizer_hpp
#define Recognizer_hpp

#define PERSON_LABEL 10 //some arbitrary label

#include "CommonHeader.h"

class Recognizer {
public :
    Recognizer(const vector<Mat> imgs, const vector<int> labels, int radius=LBPH_RADIUS, int neighbors=LBPH_NEIGHBORS,
               int grid_x=LBPH_GRID_X, int grid_y=LBPH_GRID_Y, double threshold=LBPH_THRESHOLD);
    virtual ~Recognizer();
    int recognize(const Mat &face, double &confidence) const;
    
private :
    Ptr<FaceRecognizer> _model;
    cv::Size _faceSize;
};

#endif /* Recognizer_hpp */
