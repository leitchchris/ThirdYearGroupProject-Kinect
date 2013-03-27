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
#include <fstream>

#include "libfreenect.hpp"      // For Kinect - see note for install
#include "libfreenect_sync.h"
#include <opencv2/opencv.hpp>   // Image processing - see note for install

using namespace std;
using namespace cv;

//================= Functions =======================
bool detectLum (Mat rgb_frame);
void detectFace(Mat frame);
string intToString ( int nb );
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');

//=================Global variables==============
#define darkThreshold       10
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);
bool faceDetected = false;
bool training = false;
int nbpersonIndex = 0, count2 = 0;
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
    //-----------Face reco init---------------
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 3) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }

    // Get the path to your CSV:
    string fn_haar = string(argv[1]);
    string fn_csv = string(argv[2]);

    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

//-----------------------------
    // Init here
    printf("Initialisation\n");
    Mat rgb_frame(Size(640,480),CV_8UC3);
    Mat IRimg;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
    device.startVideo();
    printf("Start video\n");

    bool lum;
    bool once = false, infra = false;
    bool alt = true;
    int countFaceD = 0;
    bool faceReco = false;


    sleep(1);       // Waste first wrong frame

    while(true)
    {
        //if(infra == false)
            device.getVideo(rgb_frame);
        
        if(!rgb_frame.empty())
        {
            // detectFace(rgb_frame);
            // imshow("RGB", rgb_frame);
            //-- 1. Detect luminosity level
            if(once == false)
            {
                lum = detectLum(rgb_frame);
                printf("detectLum : %d\n", lum);
                once = true;
            }

            //-- 2. Get RGB or IR
            if(lum == true)     // RGB
            {
                printf("GetRGB\n");
                //imshow("RGB", rgb_frame);
            }
            if(lum == false)                // Infrared
            {
                if(infra == false)
                {
                    device.stopVideo();
                    delete &device;
                }
                infra = true;

                printf("Get Infra\n");
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
                    IRimg = image;
                    //imshow( "Infrared", IRimg );
                }
            }

            //-- 3. Detect face
            // for(int i=0; i<20; i++)
            // {
                printf("detectFace\n");
                device.getVideo(rgb_frame);     //deal with Infra!!
                if(infra == false)
                {
                    //printf("Detect face...\n");
                    while(!faceDetected)
                    {
                        device.getVideo(rgb_frame);
                        detectFace(rgb_frame);
                        //printf("Face detect? %d\n", faceDetected);
                        //imshow("Detect Face", rgb_frame);
                    }     
                    //countFaceD++;
                    printf("Face Detected\n");
                    faceDetected = false;
                }    
                //else
                    //if(detectFace(IRimg))
                    //    count++;
                //sleep(1);
            //}
            // if (countFaceD >= 5)
            // {
            //     printf("Face detected\n");
            //     countFaceD = 0;
            //     faceReco = true;
            // }
            // if (countFaceD < 5)
            // {
            //     countFaceD = 0;
            //     faceReco = false;
            // }

            //-- 4. Face reco
            if(false && faceReco == true)       //If face reco otherwise training
            {
                // device.getVideo(rgb_frame);
                // // Clone the current frame:
                // Mat original = rgb_frame.clone();
                // // Convert the current frame to grayscale:
                // Mat gray;
                // cvtColor(original, gray, CV_BGR2GRAY);
                // // Find the faces in the frame:
                // vector< Rect_<int> > faces;
                // haar_cascade.detectMultiScale(gray, faces);
                // // At this point you have the position of the faces in
                // // faces. Now we'll get the faces, make a prediction and
                // // annotate it in the video. Cool or what?
                // for(int i = 0; i < faces.size(); i++) {
                //     // Process face by face:
                //     Rect face_i = faces[i];
                //     // Crop the face from the image. So simple with OpenCV C++:
                //     Mat face = gray(face_i);
                //     // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
                //     // verify this, by reading through the face recognition tutorial coming with OpenCV.
                //     // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
                //     // input data really depends on the algorithm used.
                //     //
                //     // I strongly encourage you to play around with the algorithms. See which work best
                //     // in your scenario, LBPH should always be a contender for robust face recognition.
                //     //
                //     // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
                //     // face you have just found:
                //     Mat face_resized;
                //     cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
                //     // Now perform the prediction, see how easy that is:
                //     int prediction = model->predict(face_resized);
                //     // And finally write all we've found out to the original image!
                //     // First of all draw a green rectangle around the detected face:
                //     rectangle(original, face_i, CV_RGB(0, 255,0), 1);
                //     // Create the text we will annotate the box with:
                //     string box_text = format("Prediction = %d", prediction);
                //     // Calculate the position for annotated text (make sure we don't
                //     // put illegal values in there):
                //     int pos_x = std::max(face_i.tl().x - 10, 0);
                //     int pos_y = std::max(face_i.tl().y - 10, 0);
                //     // And now put it into the image:
                //     putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                // }
                // imshow("face_recognizer", original);  
            }
            else if ( faceReco == false )
            {

                printf("faceTraining\n");
                system(("mkdir img/" + intToString(nbpersonIndex)).c_str());
                for(count2 = 0; count2 < 20; count2++)
                {
                    device.getVideo(rgb_frame);     //deal with Infra!!
                    training = true;
                    if( alt == false) //One frame on to can be more for more random faces.
                    {
                        while(!faceDetected)
                        {
                            device.getVideo(rgb_frame);
                            detectFace(rgb_frame);
                        }
                        faceDetected = false;
                        //printf("update database file\n");
                    }
                    alt = !alt;
                    printf("%d\n", count2);
                }
                system("python csv.py img/ > facerec_at_t.txt");
                nbpersonIndex++;
            } 
            
            training = false;
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

/** @function detectAndDisplay */
void detectFace( Mat frame )
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
       //printf("Eyes%d x : %f\nEyes%d y : %f \n", j, faces[i].x + eyes[j].x + eyes[j].width*0.5 , j, faces[i].y + eyes[j].y + eyes[j].height*0.5);

       //Return true only if at least one face and two eyes.
        if(j == 1)
        {
            faceDetected = true;
            if(training == true)
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
                sleep(1);
                printf ("Done\n");
                string command2 = "rm img/" + intToString(nbpersonIndex) + "/" + intToString(count2) + ".jpg";
                printf(command2.c_str());
                printf("\n");
                system(command2.c_str());    //Remove init image
                sleep(1);
            }
        }     
        else
            faceDetected = false;
    }
  }
  //printf("End detect\n");
  //faceDetected = false;
    //-- Show what you got
  //imshow( "Detect Face", frame );
}

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) 
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

string intToString ( int nb )
{
    std::string s;
    std::stringstream out;
    out << nb;
    s = out.str();
    return s;
}