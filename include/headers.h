/**
 * @file headers.h
 * @author Jose Cappelletto (cappelletto@gmail.com)
 * @brief Single collection of global libraries required in major modules
 * @version 0.2
 * @date 2020-07-03
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#ifndef _PROJECT_HEADERS_H_

#define _PROJECT_HEADERS_H_

// debug-profiling helpers for CGAL ops (bottleneck)
// #define CGAL_PROFILE
// #include <CGAL/Profile_counter.h>

///Basic C and C++ libraries
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
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>

// CUDA specific libraries
#if USE_GPU
    #include <opencv2/cudafilters.hpp>
    #include "opencv2/cudafeatures2d.hpp"
    #include "opencv2/xfeatures2d/cuda.hpp"
#endif
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

#define LO_NPART  5 //number of partitions of LO_PROTRUSION height map
#define DEFAULT_NTHREADS 12

#define DEFAULT_OUTPUT_FILE "LAD_output.tif"
#define DEFAULT_WINDOW_WIDTH 800
#define DEFAULT_WINDOW_HEIGHT 600
#define WATER_DENSITY 1025 // kg / m3
#define GRAVITY       9.81 // kg * m / s2

// #include <CGAL/Simple_cartesian.h>
// #include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kernel/global_functions.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/linear_least_squares_fitting_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// typedef CGAL::Simple_cartesian<double>  K;          // redefinition to avoid name clashing with OpenCV
typedef K::FT                           FT;       // ft
typedef K::Ray_3                        Ray;      // ray (for vertical intersection)
typedef K::Vector_3                     KVector;
typedef K::Line_3                       KLine;
typedef K::Plane_3                      KPlane;
typedef K::Point_3                      KPoint;
typedef K::Triangle_3                   KTriangle;
typedef CGAL::Polyhedron_3<K>           Polyhedron_3;


#define DEFAULT_Z_OPT  0.01  // DEFAULT value for optimal range of distance [m] along sensing axis Z (normal to the seafloor)
#define DEFAULT_Z_SUB  0.02  // DEFAULT value for suboptimal range of distance [m] along sensing axis Z (normal to the seafloor)
#define DEFAULT_G_DIAM 0.10  // DEFAULT value for diameter [m] of geotechnical sensor


#endif // _PROJECT_HEADERS_H_