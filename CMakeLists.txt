# Created by J. Cappelletto (cappelletto@gmail.com) for tiff2png module, part of Predicting seafloor landability
# Ocean Perception Lab, University of Southampton
# All rights reserved, 2019-2022

cmake_minimum_required(VERSION 3.18)  # future releases may require 3.20. Improved FindModules for GDAL, CGAL, TBB & CUDA toolkit

if(NOT DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_COLOR_MAKEFILE ON)  # For fancy colouring scheme
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options (project_options INTERFACE -fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options (project_options INTERFACE -fcolor-diagnostics)
    endif ()
endif ()

if(NOT WIN32) # Adding some colour to your life, even if you are colour-blind
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

# Define project name
project(tiff2png_project)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -mavx2") # AVX2 architecture pref.
# Additional flags for vectorized/multithread profiling and/or debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -shared-libgcc")

find_package(OpenMP)  # no alternatives provided for OpenMP as MT framework, we stick to this one
if(OPENMP_FOUND)
  message(STATUS "${BoldYellow}" "OpenMP found..." "${ColourReset}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Looking for GDAL;: https://cmake.org/cmake/help/v3.0/module/FindGDAL.html
# GDAL is used as driver for reading (and preserving) georef data
find_package(GDAL 3.0 REQUIRED) # Newest releases of GDAL provide VERSION information
if (GDAL_FOUND)
  message(STATUS "${BoldYellow}" "GDAL found: ${GDAL_VERSION}" "${ColourReset}")
  message(STATUS "    libraries: ${GDAL_LIBRARY}")
  message(STATUS "    include path: ${GDAL_INCLUDE_DIR}")
else()
  message ("GDAL library not found!")
  message ("Please install GDAL before running cmake. Quitting ...")
  return ()
endif()

find_package(Eigen3 3 REQUIRED)	# triggered some linking issues with OpenCV4.2 (sys/ROS) and user installed OpenCV 4.5

# Find OpenCV, you set the OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# OpenCV is used for image conversion from bathmetry data to grayscale image (PNG)
# Being honest, we could do it with 3.4.X but for future compatibility we should push to the latest LTS release (4.X)
find_package(OpenCV 4.2 REQUIRED
                        NO_MODULE
                        PATHS /usr/local
                        NO_DEFAULT_PATH)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# At this level, it should be useful to check minimum required version or exact version
# There are still some issues with OpenCV installations included in ROS.
# To mitigate this, please be sure you are defining the correct PATHS
message(STATUS "${BoldYellow}" "OpenCV library status:" "${ColourReset}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add external/geotiff subdirectory. We do not want to trigger its own CMake chain, so we just set the path
set (GEOTIFF_MODULE external/geotiff) # from 3rd party repository (ours, but separated for maintenance reasons)

# Greedy include of all headers
file(GLOB PROJECT_HEADERS include/*.h include/*.hpp)
include_directories(BEFORE ../include
                    include
                    ${OpenCV_INCLUDE_DIRS}
                    ${GDAL_INCLUDE_DIR}
                    ${GEOTIFF_MODULE}/include)

# Retrieve git commit information, forward it to the compilation chain
exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --abbrev=4 --dirty --always --tags"
    OUTPUT_VARIABLE GIT_INFO )
add_definitions( -DGIT_COMMIT="${GIT_INFO}" ) # Forward current git info as a single string
add_definitions( -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" ) # Forward current git info as a single string
add_definitions(-DCGAL_USE_BASIC_VIEWER)

############################ TIFF2PNG ####################
add_executable(tiff2png   src/tiff2png.cpp
                          ${GEOTIFF_MODULE}/src/geotiff.cpp
                          src/helper.cpp
                          ${PROJECT_HEADERS})

target_compile_options(tiff2png     PUBLIC -std=c++14 -mavx)
target_link_libraries(tiff2png      ${OpenCV_LIBS} ${GDAL_LIBRARY} yaml-cpp)
install (TARGETS tiff2png       DESTINATION $ENV{HOME}/bin)
# copy scripts from scripts directory to $HOME/bin directory. Using PROGRAMS guarantees that execution bit is enabled
install (PROGRAMS scripts/tiff.convert2png.sh DESTINATION $ENV{HOME}/bin)
