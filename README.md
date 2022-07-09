# tiff2png
Stand-alone module that converts geoTIFF images into 8/16 bit PNG format. Used as part of the preprocessing step for terrain feature extraction. The terrain information (topo-bathymetry) is centered by substracting its mean value and then scaled according to the user-defined parameter __max_z__. Depending on the output resolution, the maximum value will be mapped to 255 (8 bits PNG) or 65535 (16 bits PNG). Additional parameters allow defining a ROI from the input file that will be extracted and converted.

## Features
* Can read any geoTIFF topography or bathymetry map supported by GDAL
* Can export 8/18 bit PNG. Additionally, single channel (grayscale) or 3-channel (RGB) output formats are available
* User defined ROI (Region of Interest) to be exported
* Support CRS projections for Earth(WGS84) and Mars(IAU2000:49901)
* Companion parallel bash script for processing large batches 
## Installation
Start with installing the following dependencies:

* OpenCV 4.X: 
```
sudo apt install libopencv-dev
```

* GDAL 2.X, 3.X
```
sudo apt install libgdal-dev
```
___Note:___ if you have installed OpenCV of GDAL from a different source (e.g. locally compiled), you can still use it defining their corresponding installation paths when running __cmake__

Then proceed to clone this repository:

```
git clone --recurse-submodules https://github.com/cappelletto/tiff2png.git
```

Create a __build__ directory and run CMake
```
mkdir build
cd build
cmake ..
```
Finally, complete the compilation and installation (optional)

```
make -j `nproc --ignore=2`
make install
```

## How to use it

## How to contribute

## Credits


