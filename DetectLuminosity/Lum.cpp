/*
* Author			: Etienne Hocquard
* Last Modified 	: 22th February 2013	       Created  :  22th February 2013    
* File			    : OCvKinect2.cpp
* Target			: Kinect home automation project 
* Version		    : 1.0.0
* Description		: Face detection using openCV and the kinect with openni drivers
* Requires			: Kinect, openni, openCV --! haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml !--
* G++				: gcc -o Lum Lum.cpp -I/usr/local/include/libfreenect -fPIC -g -Wall `pkg-config --cflags opencv` `pkg-config --libs opencv` -L/usr/local/lib -lfreenect
* Src 				: http://openkinect.org/wiki/C%2B%2BOpenCvExample
*/

#include <iostream>
#include <stdio.h>

#include "libfreenect.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//=================Headers=======================

//=================Global variables==============
string window_name = "Capture - Face detection";

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

//================Function Main==================
int main( int argc, const char** argv )
{
	Mat rgb_frame(Size(640,480),CV_8UC3);
	Mat luminosity(Size(640,480),CV_8UC3);

	//-- 2. Read the video stream
	Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
    //printf("Device\n");
    device.startVideo();
    //printf("Start video\n");


	while( true )
	{
		device.getVideo(rgb_frame);
		//printf ("Get Video\n");
		//cv::imshow("rgb", rgb_frame);
		//-- 3. Apply the classifier to the frame
    	if(!rgb_frame.empty())
    	{ 
    		cvtColor(rgb_frame, luminosity, CV_BGR2GRAY);

			cv::Scalar avgPixelIntensity = cv::mean( luminosity );

			//prints out only .val[0] since image was grayscale
			printf("%f\n", avgPixelIntensity.val[0]);

    		/*
    		// let's quantize the hue to 30 levels
    		// and the saturation to 32 levels
    		int hbins = 30, sbins = 32;
    		int histSize[] = {hbins, sbins};
    		// hue varies from 0 to 179, see cvtColor
    		float hranges[] = { 0, 180 };
    		// saturation varies from 0 (black-gray-white) to
    		// 255 (pure spectrum color)
    		float sranges[] = { 0, 256 };
    		const float* ranges[] = { hranges, sranges };
    		MatND hist;
    		// we compute the histogram from the 0-th and 1-st channels
    		int channels[] = {0, 1};
    		calcHist( &luminosity, 2, channels, Mat(), // do not use mask
     			hist, 2, histSize, ranges,
    		    true, // the histogram is uniform
        		false );
    		double maxVal=0;
    		minMaxLoc(hist, 0, &maxVal, 0, 0);
    		//printf("Max val : %f \n", maxVal);

    		 int scale = 10;
    		 Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
    		 for( int h = 0; h < hbins; h++ )
    		    for( int s = 0; s < sbins; s++ )
    		    {
    		    		float binVal = hist.at<float>(h, s);
    		        	int intensity = cvRound(binVal*255/maxVal);
    		        	//IplImage histImgIpl = histImg;
    		        	//cvRectangle( &histImgIpl, Point(h*scale, s*scale), Point( (h+1)*scale - 1, (s+1)*scale - 1), Scalar::all(intensity), CV_FILLED );
    		    		printf("%d\n", intensity);
    		    }
    		printf("new\n\n\n\n\n");
    		namedWindow( "Source", 1 );
    		imshow( "Source", luminosity );

    		// namedWindow( "H-S Histogram", 1 );
    		// imshow( "H-S Histogram", histImg );
		*/
    	}
    	else
    	{ 
    		printf(" --(!) No captured frame -- Break!"); 
    		break; 
    	}

    	int c = waitKey(10);
    	if( (char)c == 'c' ) { break; }
    }
   return 0;
}