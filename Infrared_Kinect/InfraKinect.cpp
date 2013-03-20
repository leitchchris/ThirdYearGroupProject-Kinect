/*
* Author			: Etienne Hocquard
* Last Modified 	: 22th February 2013	       Created  :  22th February 2013    
* File			    : OCvKinect2.cpp
* Target			: Kinect home automation project 
* Version		    : 1.0.0
* Description		: Face detection using openCV and the kinect with openni drivers
* Requires			: Kinect, openni, openCV --! haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml !--
* G++				: gcc -o InfraKinect InfraKinect.cpp -I/usr/local/include/libfreenect -fPIC -g -Wall `pkg-config --cflags opencv` `pkg-config --libs opencv` -L/usr/local/lib -lfreenect
*/

#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "libfreenect.hpp"
#include "libfreenect_cv.h"
#include "libfreenect_sync.h"

using namespace std;
using namespace cv;

//=================Headers=======================
void detectAndDisplay( Mat frame, Mat framergb);

//=================Global variables==============
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

//=================Kinect device=================
class MyFreenectDevice : public Freenect::FreenectDevice {
  public:
	MyFreenectDevice(freenect_context *_ctx, int _index)
		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false),
		  depthMat(Size(640,480),CV_16UC1), rgbMat(Size(640,480),CV_8UC3,Scalar(0)), ownMat(Size(640,480),CV_8UC3,Scalar(0))
	{
		for( unsigned int i = 0 ; i < 2048 ; i++) {
			float v = i/2048.0;
			v = std::pow(v, 3)* 6;
			m_gamma[i] = v*6*256;
		}
	}
	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		std::cout << "RGB callback" << std::endl;
		m_rgb_mutex.lock();
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		rgbMat.data = rgb;
		m_new_rgb_frame = true;
		m_rgb_mutex.unlock();
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		std::cout << "Depth callback" << std::endl;
		m_depth_mutex.lock();
		uint16_t* depth = static_cast<uint16_t*>(_depth);
		depthMat.data = (uchar*) depth;
		m_new_depth_frame = true;
		m_depth_mutex.unlock();
	}

	bool getVideo(Mat& output) {
		m_rgb_mutex.lock();
		if(m_new_rgb_frame) {
			cv::cvtColor(rgbMat, output, CV_RGB2BGR);
			m_new_rgb_frame = false;
			m_rgb_mutex.unlock();
			return true;
		} else {
			m_rgb_mutex.unlock();
			return false;
		}
	}

	bool getDepth(Mat& output) {
			m_depth_mutex.lock();
			if(m_new_depth_frame) {
				depthMat.copyTo(output);
				m_new_depth_frame = false;
				m_depth_mutex.unlock();
				return true;
			} else {
				m_depth_mutex.unlock();
				return false;
			}
		}

  private:
	std::vector<uint8_t> m_buffer_depth;
	std::vector<uint8_t> m_buffer_rgb;
	std::vector<uint16_t> m_gamma;
	Mat depthMat;
	Mat rgbMat;
	Mat ownMat;
	Mutex m_rgb_mutex;
	Mutex m_depth_mutex;
	bool m_new_rgb_frame;
	bool m_new_depth_frame;
};


IplImage *GlViewColor(IplImage *depth)
{
	static IplImage *image = 0;
	if (!image) image = cvCreateImage(cvSize(640,480), 8, 3);
	unsigned char *depth_mid = (unsigned char*)(image->imageData);
	int i;
	for (i = 0; i < 640*480; i++) {
		int lb = ((short *)depth->imageData)[i] % 256;
		int ub = ((short *)depth->imageData)[i] / 256;
		switch (ub) {
			case 0:
			depth_mid[3*i+2] = 255;
			depth_mid[3*i+1] = 255-lb;
			depth_mid[3*i+0] = 255-lb;
			break;
			case 1:
			depth_mid[3*i+2] = 255;
			depth_mid[3*i+1] = lb;
			depth_mid[3*i+0] = 0;
			break;
			case 2:
			depth_mid[3*i+2] = 255-lb;
			depth_mid[3*i+1] = 255;
			depth_mid[3*i+0] = 0;
			break;
			case 3:
			depth_mid[3*i+2] = 0;
			depth_mid[3*i+1] = 255;
			depth_mid[3*i+0] = lb;
			break;
			case 4:
			depth_mid[3*i+2] = 0;
			depth_mid[3*i+1] = 255-lb;
			depth_mid[3*i+0] = 255;
			break;
			case 5:
			depth_mid[3*i+2] = 0;
			depth_mid[3*i+1] = 0;
			depth_mid[3*i+0] = 255-lb;
			break;
			default:
			depth_mid[3*i+2] = 0;
			depth_mid[3*i+1] = 0;
			depth_mid[3*i+0] = 0;
			break;
		}
	}
	return image;
}

//================Function Main==================
int main( int argc, const char** argv )
{

	while (true) {
			char *irBuffer  = (char*) malloc( 640*480*sizeof(char));
	char *irBufferTemp = 0;
    char *rgbBuffer = 0;
    uint32_t ts;
    printf("Init\n");
		/*	//-- 2. Read the video stream
	
		//Freenect::setVideoFormat(FREENECT_VIDEO_IR_8BIT);
		IplImage *image = freenect_sync_get_rgb_cv(0);
		if (!image) {
			printf("Error: Kinect not connected?\n");
			return -1;
		}
		cvCvtColor(image, image, CV_RGB2BGR);
		IplImage *depth = freenect_sync_get_depth_cv(0);
		if (!depth) {
			printf("Error: Kinect not connected?\n");
			return -1;
		}*/
		IplImage *image = 0; 
		Mat Test();
		if (!image) image = cvCreateImageHeader(cvSize(640,488), 8 , 1); 
		unsigned int timestamp;
		printf("Init 2\n");
	    if( freenect_sync_get_video((void**)&irBufferTemp,&timestamp,0,FREENECT_VIDEO_IR_8BIT)) 
	    	return NULL;
	    
	    else
	    {	printf("Get img\n");
	    	cvSetData(image, irBufferTemp, 640*3 ) ; 
	    	printf("Display\n");
			if(image != NULL)
				cvShowImage("RGB", image);
			//imwrite("Test.jpg", image);
		}
		//cvShowImage("Depth", GlViewColor(depth));
	}
	return 0;
	
	//Mat rgb_frame;
	//Mat depthMat(Size(640,480),CV_16SC1);
	//Mat depthf;

	//-- 1. Load the cascades
	// if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	// if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	/*Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
    device.setVideoFormat(FREENECT_VIDEO_IR_8BIT );
    device.startVideo();*/
    
    // while(true) { 
    // 	freenect_sync_get_video((void**)&irBufferTemp, &ts, 0, FREENECT_VIDEO_IR_8BIT);
    // 	memcpy( irBuffer, irBufferTemp, 640*480*sizeof(char));
    // 	imshow("IR",irBuffer); 
    // }
	/*device.startDepth();
	while( true )
	{
		device.getVideo(rgb_frame);
		device.getDepth(depthMat); 
		depthMat.convertTo(depthf,CV_8U,-255/4096.0,255);
		imshow("depth",rgb_frame);
        
		//-- 3. Apply the classifier to the frame
    	if(!rgb_frame.empty())
    	{ 
    		detectAndDisplay( depthf, rgb_frame );
    	}
    	else
    	{ 
    		printf(" --(!) No captured frame -- Break!"); 
    		break; 
    	}

    	int c = waitKey(10);
    	if( (char)c == 'c' ) { break; }
    }*/
   //return 0;
}

//================function detectAndDisplay================
static bool once = false;
void detectAndDisplay( Mat frame, Mat framergb )
{
	std::vector<Rect> faces;
	Mat frame_gray = frame;
	Mat frame_gray_rgb;
	cvtColor( framergb, frame_gray_rgb, CV_BGR2GRAY );
	equalizeHist( frame_gray_rgb, frame_gray_rgb );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	for( int i = 0; i < faces.size(); i++ )
	{
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		for( int j = 0; j < eyes.size(); j++ )
		{
			Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
			//Save image
       
	}
	if(once == false)
       {
          printf ("Save...");
          imwrite("img/Test.jpg", frame_gray_rgb);
          once = true;
          printf ("Done");
  		}

		}
	//-- Show what you got
	imshow( window_name, frame );
}