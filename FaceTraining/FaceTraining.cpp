  /*
* Author          : Etienne Hocquard
* Last Modified   : 21th March 2013        Created  :  21th March 2013   
* File            : FaceTraining.cpp
* Target          : Kinect home automation project 
* Version         : 1.0.0
* Description     : Detect face from image saved, call a python script with eyes position to crop and align faces images
* Requires        : openni, openCV --! haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml
* G++             : "Cmake ." + "make"
*/

 #include "libfreenect.hpp"
 // #include "opencv2/objdetect/objdetect.hpp"
 // #include "opencv2/highgui/highgui.hpp"
 // #include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

 #include <iostream>
 #include <stdio.h>
 #include <unistd.h>   // For sleep
 #include <sstream> //For int to string

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );
 string intToString ( int nb );
  bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2);

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);
 int nbimage = 0;
   int nbpersonIndex = 0;
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



 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat rgb_frame(Size(640,480),CV_8UC3,Scalar(0));
   bool alt = true;
   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Start video
   //-- 2. Read the video stream
    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
    device.startVideo();

   //-- 3. If face not recognize
   //...

   //-- 4. Create new person folder
   nbpersonIndex++;
   system(("mkdir img/" + intToString(nbpersonIndex)).c_str());

   //-- 4. Take picture while frame are different and two eyes are detect.
   printf("Start video");
  while( true )
  {
    device.getVideo(rgb_frame);
    //printf ("Get Video");
    //-- 3. Apply the classifier to the frame
      if(!rgb_frame.empty())
      { 

        if( alt == false) //One frame on to can be more for more random faces.
        {
          printf ("Detect face\n");
          detectAndDisplay(rgb_frame );

          //-- 5. Update the faces folder list.
          printf("update database file\n");
          system("python csv.py img/ > facerec_at_t.txt");
        }
        alt = !alt;
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

static int count2 = 0;
/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;


  cvtColor( frame, frame_gray, CV_BGR2GRAY );
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
       printf("Eyes%d x : %f\nEyes%d y : %f \n", j, faces[i].x + eyes[j].x + eyes[j].width*0.5 , j, faces[i].y + eyes[j].y + eyes[j].height*0.5);
       printf(count2 + "\n");
       //Delay!!
       //Crop and align face if two eyes
       if(j == 1 && count2 < 11)
       {
          string name = "img/" + intToString(nbpersonIndex) + "/" + intToString(count2) + ".jpg";
           imwrite(name, frame);
           sleep(1);

          string command;
          if ( eyes[j-1].x < eyes[j].x)
            command = "python crop.py " + intToString(nbpersonIndex) + "/" + intToString(count2) + " " + intToString(faces[i].x + eyes[j-1].x + eyes[j-1].width*0.5) + " " + intToString(faces[i].y + eyes[j-1].y + eyes[j-1].height*0.5) + " " + intToString(faces[i].x + eyes[j].x + eyes[j].width*0.5) + " " + intToString(faces[i].y + eyes[j].y + eyes[j].height*0.5) ;
          else
            command = "python crop.py " + intToString(nbpersonIndex) + "/" + intToString(count2) + " " + intToString(faces[i].x + eyes[j].x + eyes[j].width*0.5) + " " + intToString(faces[i].y + eyes[j].y + eyes[j].height*0.5) + " " + intToString(faces[i].x + eyes[j-1].x + eyes[j-1].width*0.5) + " " + intToString(faces[i].y + eyes[j-1].y + eyes[j-1].height*0.5) ;
          printf ("%s \n", command.c_str());
          system(command.c_str());
          printf ("Done\n");
          string command2 = "rm img/" + intToString(nbpersonIndex) + "/" + intToString(count2) + ".jpg";
          printf(command2.c_str());
          printf("\n");
          system(command2.c_str());    //Remove init image
          sleep(1);
          count2++;
       }
     }
  }
  //imwrite(name, frame);
  //sleep(2);
  //-- Show what you got
  imshow( window_name, frame );
 }

 string intToString ( int nb )
 {
    std::string s;
    std::stringstream out;
    out << nb;
    s = out.str();
    return s;
 }

 bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2)
{
    bool success = true;
    // check if is multi dimensional
    if(data1.dims > 2 || data2.dims > 2)
    {
      if( data1.dims != data2.dims || data1.type() != data2.type() )
      {
        return false;
      }
      for(int dim = 0; dim < data1.dims; dim++){
        if(data1.size[dim] != data2.size[dim]){
          return false;
        }
      }
    }
    else
    {
      if(data1.size() != data2.size() || data1.channels() != data2.channels() || data1.type() != data2.type()){
        return false;
      }
    }
    int nrOfElements = data1.total()*data1.elemSize1();
    //bytewise comparison of data
    int cnt = 0;
    for(cnt = 0; cnt < nrOfElements && success; cnt++)
    {
      if(data1.data[cnt] != data2.data[cnt]){
        success = false;
      }
    }
    return success;
  }