/*
* Author			: Etienne Hocquard
* Last Modified 	: 21th March 2013	       Created  :  21th March 2013   
* File			    : InfraKinect.cpp
* Target			: Kinect home automation project 
* Version		    : 1.0.0
* Description		: Infrared Face detection using openCV and the kinect with openni drivers
* Requires			: Kinect, openni, openCV --! haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml
* G++				: gcc -o InfraKinect InfraKinect.cpp -I/usr/local/include/libfreenect -fPIC -g -Wall `pkg-config --cflags opencv` `pkg-config --libs opencv` -L/usr/local/lib -lfreenect -lfreenect_cv -lfreenect_sync
*/

#include <iostream>
#include <stdio.h>
#include <sstream> //For int to string
#include <unistd.h>   // For sleep

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "libfreenect.hpp"
#include "libfreenect_cv.h"
#include "libfreenect_sync.h"

using namespace std;
using namespace cv;

//=================Headers=======================
void detectAndDisplay( Mat frame);
string intToString ( int nb );

//=================Global variables==============
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
string window_name2 = "Capture - Face detection 2";
RNG rng(12345);

//=================Kinect device=================

//================Function Main==================
int main( int argc, const char** argv )
{
	//-- 1. Load the cascades
 	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
 	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	while (true) {

		//-- 2. Get Infrared image
		char *irBufferTemp = 0;
		IplImage* image = 0; 
		if (!image) image = cvCreateImageHeader(cvSize(640,488), 8 , 1); 
		unsigned int timestamp;

		if( freenect_sync_get_video((void**)&irBufferTemp,&timestamp,0,FREENECT_VIDEO_IR_8BIT)) 
			return NULL;
		else
		{	
			cvSetData(image, irBufferTemp, 640*1 ) ; 
			Mat IRimg(image);

			//-- 3. Detect face and display
			detectAndDisplay( IRimg );
			int c = waitKey(10);
    		if( (char)c == 'c' ) { break; }
		}
	}
	return 0;
	
}

//================function detectAndDisplay================
static int count2 = 0;
/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray = frame;

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( int i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( int j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       //circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );

       //Save image if only least one face and two eyes.
       if(count2 <= 10 && j == 1)
       {
          printf ("Save...\n");
          string name = "img/" + intToString(count2) + ".jpg";
          imwrite(name, frame);
          //sleep(3);
          count2++;

          // string command = "python crop.py " + intToString(count2) + " " + intToString(eyes[j-1].x) + " " + intToString(eyes[j-1].y) + " " + intToString(eyes[j].x) + " " + intToString(eyes[j].y) ;
          // printf ("%s \n", command.c_str());
          // system(command.c_str());

          printf ("Done\n");
       }
     }
  }
  
  //-- Show what you got
  imshow( window_name, frame );
  //imshow( window_name2, frame_gray );
 }

  string intToString ( int nb )
 {
    std::string s;
    std::stringstream out;
    out << nb;
    s = out.str();
    return s;
 }