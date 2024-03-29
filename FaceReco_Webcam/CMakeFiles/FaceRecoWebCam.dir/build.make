# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tinou/Kinect/FaceReco_Webcam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tinou/Kinect/FaceReco_Webcam

# Include any dependencies generated for this target.
include CMakeFiles/FaceRecoWebCam.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceRecoWebCam.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceRecoWebCam.dir/flags.make

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o: CMakeFiles/FaceRecoWebCam.dir/flags.make
CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o: FaceRecoWebCam.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/tinou/Kinect/FaceReco_Webcam/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o -c /home/tinou/Kinect/FaceReco_Webcam/FaceRecoWebCam.cpp

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/tinou/Kinect/FaceReco_Webcam/FaceRecoWebCam.cpp > CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.i

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/tinou/Kinect/FaceReco_Webcam/FaceRecoWebCam.cpp -o CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.s

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.requires:
.PHONY : CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.requires

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.provides: CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.requires
	$(MAKE) -f CMakeFiles/FaceRecoWebCam.dir/build.make CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.provides.build
.PHONY : CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.provides

CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.provides.build: CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o

# Object files for target FaceRecoWebCam
FaceRecoWebCam_OBJECTS = \
"CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o"

# External object files for target FaceRecoWebCam
FaceRecoWebCam_EXTERNAL_OBJECTS =

FaceRecoWebCam: CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o
FaceRecoWebCam: CMakeFiles/FaceRecoWebCam.dir/build.make
FaceRecoWebCam: /usr/local/lib/libopencv_calib3d.so
FaceRecoWebCam: /usr/local/lib/libopencv_contrib.so
FaceRecoWebCam: /usr/local/lib/libopencv_core.so
FaceRecoWebCam: /usr/local/lib/libopencv_features2d.so
FaceRecoWebCam: /usr/local/lib/libopencv_flann.so
FaceRecoWebCam: /usr/local/lib/libopencv_gpu.so
FaceRecoWebCam: /usr/local/lib/libopencv_highgui.so
FaceRecoWebCam: /usr/local/lib/libopencv_imgproc.so
FaceRecoWebCam: /usr/local/lib/libopencv_legacy.so
FaceRecoWebCam: /usr/local/lib/libopencv_ml.so
FaceRecoWebCam: /usr/local/lib/libopencv_nonfree.so
FaceRecoWebCam: /usr/local/lib/libopencv_objdetect.so
FaceRecoWebCam: /usr/local/lib/libopencv_photo.so
FaceRecoWebCam: /usr/local/lib/libopencv_softcascade.so
FaceRecoWebCam: /usr/local/lib/libopencv_stitching.so
FaceRecoWebCam: /usr/local/lib/libopencv_ts.so
FaceRecoWebCam: /usr/local/lib/libopencv_video.so
FaceRecoWebCam: /usr/local/lib/libopencv_videostab.so
FaceRecoWebCam: CMakeFiles/FaceRecoWebCam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable FaceRecoWebCam"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceRecoWebCam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceRecoWebCam.dir/build: FaceRecoWebCam
.PHONY : CMakeFiles/FaceRecoWebCam.dir/build

CMakeFiles/FaceRecoWebCam.dir/requires: CMakeFiles/FaceRecoWebCam.dir/FaceRecoWebCam.cpp.o.requires
.PHONY : CMakeFiles/FaceRecoWebCam.dir/requires

CMakeFiles/FaceRecoWebCam.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceRecoWebCam.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceRecoWebCam.dir/clean

CMakeFiles/FaceRecoWebCam.dir/depend:
	cd /home/tinou/Kinect/FaceReco_Webcam && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tinou/Kinect/FaceReco_Webcam /home/tinou/Kinect/FaceReco_Webcam /home/tinou/Kinect/FaceReco_Webcam /home/tinou/Kinect/FaceReco_Webcam /home/tinou/Kinect/FaceReco_Webcam/CMakeFiles/FaceRecoWebCam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceRecoWebCam.dir/depend

