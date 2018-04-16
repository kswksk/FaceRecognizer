//
//  main.cpp
//  SGFaceRecognizer
//
//  Created by 50017516 on 2018. 4. 9..
//  Copyright © 2018년 50017516. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <libproc.h>

#include "CommonHeader.h"
#include "FaceDetector.hpp"
#include "FaceEyeDetector.hpp"
#include "Recognizer.hpp"

// #define TRADING_MAKE_IMG

#define DEFAULT_CASCADE_PATH "resources/cascades/haarcascade_frontalface_default.xml"
#define ORIGINALS_LIST "resources/trading_raw/list"
#define OUTPUT_DIR "resources/trading_face"
#define OUTPUT_LIST "resources/trading_face/list"

// KIMSG 얼굴검출
#define KIMSG_ORIGINALS_LIST "resources/kimsg_raw/list"
#define KIMSG_OUTPUT_DIR "resources/kimsg_raw_trading"
#define KIMSG_OUTPUT_LIST "resources/kimsg_raw_trading/list"

#define FACE_SIZE cv::Size(150,150)

// const double DESIRED_LEFT_EYE_X = 0.16;     // Controls how much of the face is visible after preprocessing.
// const double DESIRED_LEFT_EYE_Y = 0.14;

#define WINDOW_NAME "KimSeongGu Koea Univ"
const int DESIRED_CAMERA_WIDTH = 960;
const int DESIRED_CAMERA_HEIGHT = 540;

bool isTrain = true;
string executePath;

string getCurrentPath() {
    int ret;
    pid_t pid;
    char pathbuf[PROC_PIDPATHINFO_MAXSIZE];
    
    pid = getpid();
    ret = proc_pidpath (pid, pathbuf, sizeof(pathbuf));
    if ( ret <= 0 ) {
        fprintf(stderr, "PID %d: proc_pidpath ();\n", pid);
        fprintf(stderr, "    %s\n", strerror(errno));
    } else {
        // printf("proc %d: %s\n", pid, pathbuf);
        string path = string(pathbuf);
        size_t pos = path.find_last_of('/');
        path = path.substr(0, pos);
        return path;
    }
    
    return string("");
}

// 이미지에서 얼굴영역을 추출한다.
void trainImage(string org_list_path, string out_list_path, string out_dir) {
    string cascadePath = format("%s/%s", executePath.c_str(), DEFAULT_CASCADE_PATH);
    FaceDetector fd(cascadePath, 1.2, 15);
    FaceEyeDetector eyeFd(executePath);
    
    vector<Mat> raw_faces;
    
    ofstream out_list(org_list_path);
    ifstream file(out_list_path);
    
    string path;
    while (getline(file, path)) {
        if (path.compare("") != 0)
            raw_faces.push_back(imread(path));
    }
    
    int img_c = 0;
    for (vector<Mat>::const_iterator raw_img = raw_faces.begin(); raw_img != raw_faces.end(); raw_img++) {
        vector<cv::Rect> faces;
        fd.findFacesInImage(*raw_img, faces);
        
        for (vector<cv::Rect>::const_iterator face = faces.begin(); face != faces.end(); face++) {
            int edge_size = max(face->width, face->height);
            int x = face->x;
            int y = face->y;
            int w = edge_size;
            int h = edge_size;
            
            cv::Rect square(x, y, w, h);
            Mat face_img = (*raw_img)(square);
            
            // 이미지 사이즈 조정
            cv::resize(face_img, face_img, FACE_SIZE);
            
            vector<cv::Rect> eyes;
            eyeFd.findEyeInFace(face_img, eyes);
            
            if (eyes.size() == 2) {
                Point2f eyesCenter;
                // 눈 중간지점
                cv::Rect leftEye = eyes.at(0);
                cv::Rect rightEye = eyes.at(1);
                eyesCenter.x = (leftEye.x+(leftEye.width*0.5f) + rightEye.x+(rightEye.width*0.5f)) * 0.5f;
                eyesCenter.y = (leftEye.y+(leftEye.height*0.5f) + rightEye.y+(rightEye.height*0.5f)) * 0.5f;
                
                /*
                CvPoint center;
                int radius;
                center.x = cvRound(leftEye.x + (leftEye.width*0.5));
                center.y = cvRound(leftEye.y + leftEye.height*0.5);
                radius = cvRound(leftEye.width + leftEye.height)*0.25;
                circle(face_img, center, radius, CV_RGB(0,255,0), 3, 8, 0 );
                
                center.x = cvRound(rightEye.x + (rightEye.width*0.5));
                center.y = cvRound(rightEye.y + rightEye.height*0.5);
                radius = cvRound(rightEye.width + rightEye.height)*0.25;
                circle(face_img, center, radius, CV_RGB(0,0,255), 3, 8, 0 );
                
                imshow("test", face_img);
                cvWaitKey();
                */
                
                // 눈 각도 계산 후 회전
                double dx = (rightEye.x - leftEye.x);
                double dy = (rightEye.y - leftEye.y);
                double angle = atan2(dy, dx) * 180.0/CV_PI;
                Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);
                Mat warped = Mat(FACE_SIZE.width, FACE_SIZE.height, CV_8U, Scalar(128));
                warpAffine(face_img, warped, rot_mat, warped.size());
                
                //imshow("test", warped);
                //cvWaitKey();
                
                // write to disk:
                string face_path = format("%s/%d.jpg", out_dir.c_str(), img_c++);
                imwrite(face_path, warped);
                out_list << face_path << endl;
            }
        }
    }
    
    out_list.close();
}

void read_recog_data(string argPath, vector<Mat> &training_set, vector<int> &training_label, int label) {
    ifstream file(argPath);
    string path;
    while (getline(file, path)) {
        path = format("%s/%s", executePath.c_str(), path.c_str());
        
        cout << " load image : " << path << endl;
        
        training_set.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
        training_label.push_back(label);
    }
}

// 학습 후
void recognition(string samplePath) {
    vector<Mat> training_set;
    vector<int> training_label;
    
    string output_list = format("%s/%s", executePath.c_str(), OUTPUT_LIST);
    read_recog_data(output_list, training_set, training_label, 10);
    output_list = format("%s/%s", executePath.c_str(), KIMSG_OUTPUT_LIST);
    read_recog_data(output_list, training_set, training_label, 99);
    
    Recognizer rec(training_set, training_label);
    vector<cv::Rect> faces;
    string cascadePath = format("%s/%s", executePath.c_str(), DEFAULT_CASCADE_PATH);
    FaceDetector det(cascadePath, 1.2, 15);
    
    Mat sample_img = imread(samplePath);
    det.findFacesInImage(sample_img, faces);
    
    imshow("test", sample_img);
    cvWaitKey();
    
    for (vector<cv::Rect>::const_iterator face = faces.begin(); face != faces.end(); face++) {
        //Scalar color = NO_MATCH_COLOR;
        Mat face_img = sample_img(*face);
        
        double confidence = 0;
        int label = -1;
        label = rec.recognize(face_img, confidence);
        
        cout << "label : " << label << ", confidence : " << confidence << endl;
        
        imshow("test", face_img);
        cvWaitKey();
    }
}

void recognitionVideo() {
    // 초기화
    vector<Mat> training_set;
    vector<int> training_label;
    
    string labelNames[] = {"kimsg", "obama"};
    
    string output_list = format("%s/%s", executePath.c_str(), OUTPUT_LIST);
    read_recog_data(output_list, training_set, training_label, 1);
    output_list = format("%s/%s", executePath.c_str(), KIMSG_OUTPUT_LIST);
    read_recog_data(output_list, training_set, training_label, 0);
    
    Recognizer rec(training_set, training_label);
    vector<cv::Rect> faces;
    string cascadePath = format("%s/%s", executePath.c_str(), DEFAULT_CASCADE_PATH);
    FaceDetector det(cascadePath, 1.2, 15);
    
    namedWindow(WINDOW_NAME);
    
    CvCapture *camera = cvCreateCameraCapture(CV_CAP_ANY);
    cvSetCaptureProperty(camera,CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    cvSetCaptureProperty(camera,CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
    
    IplImage *frame;
    while((frame = cvQueryFrame(camera))) {
        // 좌우를 반전시킨다.
        IplImage *  flipImg    = cvCreateImage(cvSize (frame->width, frame->height), IPL_DEPTH_8U, 3);
        cvFlip (frame, flipImg, 1);
        
        Mat frameMat = cvarrToMat(flipImg);
        det.findFacesInImage(frameMat, faces);
        
        for (vector<cv::Rect>::const_iterator face = faces.begin(); face != faces.end(); face++) {
            //Scalar color = NO_MATCH_COLOR;
            Mat face_img = frameMat(*face);
            double confidence = 0;
            int label = -1;
            label = rec.recognize(face_img, confidence);
            
            Scalar color = CV_RGB(255,0,0);
            if (label >= 0) {
                color = CV_RGB(0,255,0);
            }
            
            rectangle(frameMat, cvPoint(face->x, face->y), cvPoint(face->x+face->width, face->y+face->height), color);
            
            putText(frameMat, labelNames[label], cvPoint(face->x, face->y-8), FONT, SCALE_TITLE, FONT_COLOR, THICKNESS_TITLE, LINE_TYPE);
        }
        
        imshow(WINDOW_NAME, frameMat);
        int key = cvWaitKey (100);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }
    
    cvReleaseCapture(&camera);
}

int main(int argc, const char * argv[]) {
    executePath = getCurrentPath();
    
#ifdef TRADING_MAKE_IMG
    trainImage(
        format("%s/%s", executePath.c_str(), OUTPUT_LIST),
        format("%s/%s", executePath.c_str(), ORIGINALS_LIST),
        format("%s", OUTPUT_DIR)
    );
    trainImage(
       format("%s/%s", executePath.c_str(), KIMSG_OUTPUT_LIST),
       format("%s/%s", executePath.c_str(), KIMSG_ORIGINALS_LIST),
       format("%s", KIMSG_OUTPUT_DIR)
   );
#endif

#ifndef TRADING_MAKE_IMG
    // recognition(format("%s/%s", executePath.c_str(), "resources/sample_img/IMG_0138.JPG"));
    // recognition(format("%s/%s", executePath.c_str(), "resources/sample_img/obama_15.JPG"));
    recognitionVideo();
#endif
    
    return 0;
}
