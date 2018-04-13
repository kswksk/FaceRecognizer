//
//  CommonHeader.h
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#ifndef CommonHeader_h
#define CommonHeader_h

#include <iostream>
#include <CoreFoundation/CoreFoundation.h>
#include <cassert>

// Example showing how to read and write images
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/face.hpp>

using namespace cv;
using namespace cv::ml;
using namespace cv::face;
using namespace std;

/** Flags: **/
#define SHOW_OUTPUT
#define WRITE_OUTPUT
#define WRITE_CSV

/** Inputs: **/
#define CASCADE_PATH  "cascades/haarcascade_frontalface_default.xml"
#define IN_VID        "input_vid/031213_POTUS_ExportCouncil_HD.mp4"
#define TRAINING_LIST "obama_faces/list"

/** Input video: **/
#define START_FRAME  2400
#define END_FRAME    4800
#define FRAMES_DELTA -1

/** Outputs: **/
#define OUT_VID  "1.avi"
#define CSV_FILE "1.csv"

/** Output video: **/
#define OUT_FPS    15
#define OUT_FOURCC CV_FOURCC('M','J','P','G') //codec

/** Colors, fonts, lines... **/
#define NO_MATCH_COLOR    Scalar(0,0,255) //red
#define MATCH_COLOR       Scalar(0,255,0) //green
#define FACE_RADIUS_RATIO 0.75
#define CIRCLE_THICKNESS  2.5
#define LINE_TYPE         CV_AA
#define FONT              FONT_HERSHEY_PLAIN
#define FONT_COLOR        Scalar(255,255,255)
#define THICKNESS_TITLE   1.9
#define SCALE_TITLE       1.9
#define POS_TITLE         cvPoint(10, 30)
#define THICKNESS_LINK    1.6
#define SCALE_LINK        1.3
#define POS_LINK          cvPoint(10, 55)

/** Face detector: **/
#define DET_SCALE_FACTOR   1.01
#define DET_MIN_NEIGHBORS  40
//#define DET_MIN_SIZE_RATIO 0.06
//#define DET_MAX_SIZE_RATIO 0.18

/** LBPH face recognizer: **/
#define LBPH_RADIUS    3
#define LBPH_NEIGHBORS 8
#define LBPH_GRID_X    8
#define LBPH_GRID_Y    8
#define LBPH_THRESHOLD 180.0

#endif /* CommonHeader_h */
