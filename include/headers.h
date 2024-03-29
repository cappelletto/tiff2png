/**
 * @file headers.h
 * @author Jose Cappelletto (cappelletto@gmail.com). University of Southampton, UK
 * @brief Single collection of libraries required for this project
 * @version 0.3
 * @date 2020-07-03
 *
 * @copyright Copyright (c) 2020-2022
 *
 */
#ifndef _PROJECT_HEADERS_H_

#define _PROJECT_HEADERS_H_

/// Basic C and C++ libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
/// OpenCV libraries. May need review for the final release
#include <opencv2/core.hpp>
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "geotiff.hpp"

// escape based colour codes for console output
const std::string red("\033[1;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string purple("\033[1;35m");
const std::string cyan("\033[1;36m");

const std::string light_red("\033[0;31m");
const std::string light_green("\033[0;32m");
const std::string light_yellow("\033[0;33m");
const std::string light_blue("\033[0;34m");
const std::string light_purple("\033[0;35m");
const std::string light_cyan("\033[0;36m");

const std::string reset("\033[0m");
const std::string highlight("\033[30;43m");

#define LO_NPART 5 // number of partitions of LO_PROTRUSION height map
#define DEFAULT_NTHREADS 12

#define DEFAULT_OUTPUT_FILE "LAD_output.tif"
#define DEFAULT_WINDOW_WIDTH 800
#define DEFAULT_WINDOW_HEIGHT 600

#define T2P_GRAYSCALE 1
#define T2P_RGB 3

#define T2P_BPP8 8
#define T2P_BPP16 16

#endif // _PROJECT_HEADERS_H_