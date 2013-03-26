/*
* Author          : Etienne Hocquard
* Last Modified   : 26th March 2013        Created  :  26th March 2013   
* File            : Kinect.cpp
* Target          : Kinect home automation project 
* Version         : 1.0.0
* Description     : Face identification for the project "home automation" - 3rd year Edinburh Napier University
* Requires        : Kinect, freenect, openCV, haarcascade_eye_tree_eyeglasses.xml + haarcascade_frontalface_alt.xml
* G++             : To define
*/

//================= Include =======================
#include <iostream>
#include <stdio.h>
#include <unistd.h>             // For sleep thread
#include <sstream>              // For int to string

#include "libfreenect.hpp"      // For Kinect - see note for install
#include <opencv2/opencv.hpp>   // Image processing - see note for install
