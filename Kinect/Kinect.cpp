/*
* Author          : Etienne Hocquard
* Last Modified   : 26th March 2013        Created  :  26th March 2013   
* File            : Kinect.cpp
* Target          : Kinect home automation project 
* Version         : 1.0.0
* Description     : Face identification for the project "home automation" - 3rd year Edinburh Napier University
* Requires        : Kinect, freenect, openCV, haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml
* G++             : gcc -o Kinect Kinect.cpp -I/usr/local/include/libfreenect -fPIC -g -Wall `pkg-config --cflags opencv` `pkg-config --libs opencv` -L/usr/local/lib -lfreenect -lfreenect_sync
*/

//================= Include =======================
#include <iostream>
#include <stdio.h>
#include <unistd.h>             // For sleep thread
#include <sstream>              // For int to string

#include "libfreenect.hpp"      // For Kinect - see note for install
#include "libfreenect_sync.h"
#include <opencv2/opencv.hpp>   // Image processing - see note for install

using namespace std;
using namespace cv;

//================= Functions =======================
bool detectLum (Mat rgb_frame);

//=================Global variables==============
#define darkThreshold       10
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

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
    ~MyFreenectDevice()
    {
        //freenect_close_device(0);
        //freenect_shutdown(_ctx);
    }
    // Do not call directly even in child
    void VideoCallback(void* _rgb, uint32_t timestamp) {
        //std::cout << "RGB callback" << std::endl;
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
    // Init here
    printf("Initialisation\n");
    Mat rgb_frame(Size(640,480),CV_8UC3,Scalar(0));

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
    device.startVideo();
    printf("Start video\n");

    bool lum;
    bool once = false, once2 = false;

    sleep(1);       // Waste first wrong frame

    while(true)
    {
        if(once2 == false)
            device.getVideo(rgb_frame);
        
        if(!rgb_frame.empty())
        {
            //-- 1. Detect luminosity level
            if(once == false)
            {
                lum = detectLum(rgb_frame);
                printf("%d\n", lum);
                once = true;
            }
            //-- 2. Get RGB or IR
            if(lum == true)     // RGB
            {

            }
            else                // Infrared
            {
                if(once2 == false)
                {
                    device.stopVideo();
                    delete &device;
                }
                once2 = true;

                //printf("Get Infra\n");
                char *irBufferTemp = 0;
                IplImage* image = 0; 
                if (!image) image = cvCreateImageHeader(cvSize(640,488), 8 , 1); 
                unsigned int timestamp;
                if( freenect_sync_get_video((void**)&irBufferTemp,&timestamp,0,FREENECT_VIDEO_IR_8BIT)) 
                    return NULL;
                else
                {   
                    //printf("Display\n");
                    cvSetData(image, irBufferTemp, 640*1 ) ; 
                    Mat IRimg(image);
                    imshow( "Infrared", IRimg );
                }
            }
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

bool detectLum (Mat rgb_frame)
{
    Mat gray_frame(Size(640,480),CV_8UC3);

    cvtColor(rgb_frame, gray_frame, CV_BGR2GRAY);
    cv::Scalar avgPixelIntensity = cv::mean( gray_frame );

    //printf("Lum level : %f\n", avgPixelIntensity.val[0]);

    if( avgPixelIntensity.val[0] < darkThreshold )
        return false;
    else
        return true;
}