# Created by J. Cappelletto (cappelletto@gmail.com) for tiff2png module, part of Predicting seafloor landability
# Ocean Perception Lab, University of Southampton
# 2019-2021

cmake_minimum_required(VERSION 3.14)

if(NOT DEFINED CMAKE_BUILD_TYPE) # Use Release mode as default (prioritize performance)
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

# Define project name
project(tiff2png_project)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -mavx2") # AVX2 architecture pref.
# Additional flags for vectorized/multithread profiling and/or debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -shared-libgcc")

# WARNING: CGAL / QT5 setup required this preproc
# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTORCC ON)
# set(CMAKE_AUTOUIC ON)

find_package(OpenMP)  # no alternatives provided for OpenMP as MT framework, we stick to this one
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Looking for GDAL;: https://cmake.org/cmake/help/v3.0/module/FindGDAL.html
# GDAL is used as driver for reading (and preserving) georef data
find_package(GDAL 3.0 REQUIRED) # Newest releases of GDAL provide VERSION information
if (GDAL_FOUND)
  message(STATUS "GDAL found: ${GDAL_VERSION}")
  message(STATUS "    libraries: ${GDAL_LIBRARY}")
  message(STATUS "    include path: ${GDAL_INCLUDE_DIR}")
else()
  message ("GDAL library not found!")
  message ("Please install GDAL before running cmake. Quitting ...")
  return ()
endif()

find_package(Eigen3 3 REQUIRED)	# triggered some linking issues with OpenCV4.2 (sys/ROS) and user installed OpenCV 4.5

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# OpenCV used for image conversion from bathmetry data to grayscale image (PNG)
# Being honest, we could do it with 3.4.X but for future compatibility we should push to the latest LTS release (4.2)
find_package(OpenCV 4.2 REQUIRED
                        NO_MODULE
                        PATHS /usr/local
                        NO_DEFAULT_PATH)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
# At this level, it should be useful to check minimum required version or exact version
# So far, there are still some issues with OpenCV installations included in ROS
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Cross your fingers
# find_package(CUDA)

# if(CUDA_FOUND)
#     INCLUDE(FindCUDA)
#   # If the package has been found, several variables will
#   # be set, you can find the full list with descriptions
#   # in the CUDAConfig.cmake file.
#   # Print some message showing some of them
#   message(STATUS "CUDA library status:")
#   message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
#   message(STATUS "    version: ${CUDA_VERSION}")
#   message(STATUS "    libraries: ${CUDA_LIBRARIES}")
#   message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
# endif(CUDA_FOUND)

# #find_package(CGAL REQUIRED COMPONENTS Qt5 Core)
# #CGAL_Qt5 is needed for the drawing.
# find_package(CGAL REQUIRED OPTIONAL_COMPONENTS Qt5)

# if(CGAL_Qt5_FOUND)  #required to use basic_viewer
#   add_definitions(-DCGAL_USE_BASIC_VIEWER -DQT_NO_KEYWORDS)
# endif()

# if ( NOT CGAL_FOUND )
#   message(STATUS "This project requires the CGAL library, and will not be compiled.")
#   return()  
# endif()

# message(STATUS "CGAL library status:")
# message(STATUS "    libraries: ${CGAL_LIBRARY}")
# message(STATUS "    include path: ${CGAL_INCLUDE_DIR}")

# include helper file
# include( ${CGAL_USE_FILE} )

#find_package(PCL REQUIRED COMPONENTS common io)

# find_package( Boost REQUIRED ) # Boost and its components, required by CGAL
# if ( NOT Boost_FOUND )
#   message(STATUS "Required Boost library is missing. Project will not be compiled.")
#   return()  
# endif()

# Add submodule as external/geotiff subdirectory. But we do not want to trigger its own CMake chain, so we just set the path
#add_subdirectory(external/geotiff)
set (GEOTIFF_MODULE external/geotiff) # from 3rd party repository (ours, but separated for maintenance reasons)

# Greedy include of all headers
file(GLOB PROJECT_HEADERS include/*.h include/*.hpp)
include_directories(BEFORE ../include
                    include
                    ${OpenCV_INCLUDE_DIRS}
                    ${GDAL_INCLUDE_DIR}
                    ${GEOTIFF_MODULE}/include)
 #                   ${PCL_INCLUDE_DIRS})

# Retrieve git commit information, forward it to compilation chain
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
                          # src/lad_core.cpp
                          # src/lad_layer.cpp
                          # src/lad_processing.cpp
                          # src/lad_thread.cpp
                          # src/lad_config.cpp
                          ${PROJECT_HEADERS})


target_compile_options(tiff2png     PUBLIC -std=c++14 -mavx)
target_link_libraries(tiff2png      ${OpenCV_LIBS} ${GDAL_LIBRARY} yaml-cpp)
install (TARGETS tiff2png       DESTINATION $ENV{HOME}/bin)
