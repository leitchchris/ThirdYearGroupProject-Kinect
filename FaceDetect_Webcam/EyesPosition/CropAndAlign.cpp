 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>
 #include <unistd.h>   // For sleep
 #include <sstream> //For int to string

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );
 string intToString ( int nb );

 /** Global variables */
 String face_cascade_name = "../haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "../haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   frame = imread("../img/2.jpg");
   detectAndDisplay(frame);
 }

static int count2 = 2;
/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(64, 48) );

  for( int i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(64, 48) );

    for( int j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
       printf("Eyes%d x : %d\nEyes%d y : %d \n", j, eyes[j].x, j, eyes[j].y);
       //Delay!!
       //Crop and align face
       if(count2 <= 10 && j > 0 && j < 3)
       {
          string command = "python ../crop.py " + intToString(count2) + " " + intToString(eyes[j-1].x) + " " + intToString(eyes[j-1].y) + " " + intToString(eyes[j].x) + " " + intToString(eyes[j].y) ;
          printf ("%s \n", command.c_str());
          system(command.c_str());
          //sleep(1);
          printf ("Done\n");
       }
     }
  }
  string name = "../img/" + intToString(count2) + "x.jpg";
  imwrite(name, frame);
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