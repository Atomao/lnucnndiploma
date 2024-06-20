# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build

# Include any dependencies generated for this target.
include CMakeFiles/Yolov8CPPInference.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Yolov8CPPInference.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Yolov8CPPInference.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Yolov8CPPInference.dir/flags.make

CMakeFiles/Yolov8CPPInference.dir/main.cpp.o: CMakeFiles/Yolov8CPPInference.dir/flags.make
CMakeFiles/Yolov8CPPInference.dir/main.cpp.o: /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/main.cpp
CMakeFiles/Yolov8CPPInference.dir/main.cpp.o: CMakeFiles/Yolov8CPPInference.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Yolov8CPPInference.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Yolov8CPPInference.dir/main.cpp.o -MF CMakeFiles/Yolov8CPPInference.dir/main.cpp.o.d -o CMakeFiles/Yolov8CPPInference.dir/main.cpp.o -c /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/main.cpp

CMakeFiles/Yolov8CPPInference.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Yolov8CPPInference.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/main.cpp > CMakeFiles/Yolov8CPPInference.dir/main.cpp.i

CMakeFiles/Yolov8CPPInference.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Yolov8CPPInference.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/main.cpp -o CMakeFiles/Yolov8CPPInference.dir/main.cpp.s

CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o: CMakeFiles/Yolov8CPPInference.dir/flags.make
CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o: /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/inference.cpp
CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o: CMakeFiles/Yolov8CPPInference.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o -MF CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o.d -o CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o -c /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/inference.cpp

CMakeFiles/Yolov8CPPInference.dir/inference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Yolov8CPPInference.dir/inference.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/inference.cpp > CMakeFiles/Yolov8CPPInference.dir/inference.cpp.i

CMakeFiles/Yolov8CPPInference.dir/inference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Yolov8CPPInference.dir/inference.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/inference.cpp -o CMakeFiles/Yolov8CPPInference.dir/inference.cpp.s

# Object files for target Yolov8CPPInference
Yolov8CPPInference_OBJECTS = \
"CMakeFiles/Yolov8CPPInference.dir/main.cpp.o" \
"CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o"

# External object files for target Yolov8CPPInference
Yolov8CPPInference_EXTERNAL_OBJECTS =

Yolov8CPPInference: CMakeFiles/Yolov8CPPInference.dir/main.cpp.o
Yolov8CPPInference: CMakeFiles/Yolov8CPPInference.dir/inference.cpp.o
Yolov8CPPInference: CMakeFiles/Yolov8CPPInference.dir/build.make
Yolov8CPPInference: /usr/local/lib/libopencv_gapi.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_highgui.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_ml.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_objdetect.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_photo.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_stitching.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_video.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_videoio.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_dnn.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_calib3d.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_features2d.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_flann.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_imgproc.so.4.9.0
Yolov8CPPInference: /usr/local/lib/libopencv_core.so.4.9.0
Yolov8CPPInference: CMakeFiles/Yolov8CPPInference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Yolov8CPPInference"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Yolov8CPPInference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Yolov8CPPInference.dir/build: Yolov8CPPInference
.PHONY : CMakeFiles/Yolov8CPPInference.dir/build

CMakeFiles/Yolov8CPPInference.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Yolov8CPPInference.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Yolov8CPPInference.dir/clean

CMakeFiles/Yolov8CPPInference.dir/depend:
	cd /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build /home/pi/exp/exp1/src/opencv_cpp/ultralytics/examples/YOLOv8-CPP-Inference/build/CMakeFiles/Yolov8CPPInference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Yolov8CPPInference.dir/depend
