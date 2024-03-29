/**
 * @file tif2png.cpp
 * @author Jose Cappelletto (cappelletto@gmail.com / j.cappelletto@soton.ac.uk)
 * @brief geoTIFF to PNG converter. Part of the data preparation pipeline to generate the PNG training dataset for LG Autoencoder
 *        and Bayesian Neural Network inference framework
 * @version 0.2
 * @date 2020-11-09
 *
 * @copyright Copyright (c) 2020
 *
 */
#include "headers.h"
#include "helper.h"

#include "options.h"
#include "geotiff.hpp" // Geotiff class definitions
#include <limits>

using namespace std;
using namespace cv;

#define COPY_ONLY_MODE 3

#define NO_ERROR 0
#define ERR_ARGUMENT -2
logger::ConsoleOutput logc;

/*!
    @fn     int main(int argc, char* argv[])
    @brief  Main function
*/
int main(int argc, char *argv[])
{
    int retval = initParser(argc, argv); // initial argument validation, populates arg parsing structure args
    if (retval != 0)                     // some error ocurred, we have been signaled to stop
        return retval;
    std::ostringstream s;
    int verbosity = 0; // default: no verbose output
    if (argVerbose)
        verbosity = args::get(argVerbose); // input file is mandatory positional argument. Overrides any definition in configuration.yaml
    // Input file priority: must be defined either by the config.yaml or --input argument
    string inputFileName = "";  // command arg or config defined
    string outputFileName = ""; // command arg or config defined
    string outputTIFF = "";     // command arg or config defined
    // Mandatory arguments
    if (argInput)
        inputFileName = args::get(argInput); // input file is mandatory positional argument.
    if (inputFileName.empty())
    { // not defined as command line argument? let's use config.yaml definition
        logc.error("main", "Input file missing. Please define it using --input='filename'");
        return -1;
    }
    if (argOutput)
        outputFileName = args::get(argOutput); // input file is mandatory positional argument
    if (outputFileName.empty())
    {
        logc.error("main", "Output file missing. Please define it using --output='filename'");
        return -1;
    }
    // if (argExportTiff) outputTIFF = args::get(argExportTiff); //extra geotiff copy to be exported

    // Optional arguments
    // validity threshold. Default pass all (th=0)
    double validThreshold = 0.0; // default, we export any image regardless the proportion of valid pixels
    if (argValidThreshold)
        validThreshold = args::get(argValidThreshold);
    // let's validate
    if (validThreshold < 0.0 || validThreshold > 1.0)
    {
        s << "Invalid value for validThreshold [" << red << validThreshold << reset << "]. Valid range [0.0, 1.0]. Check --valid_th argument";
        logc.error("main", s);
        return -1;
    }
    // rotation, default = 0 degress
    double rotationAngle = 0;
    if (argRotation)
        rotationAngle = args::get(argRotation); // any value is valid. No validation is required
    // max depth range (both positive and negative), default = +1.0
    double maxDepth = 1.0;
    if (argZMax)
        maxDepth = args::get(argZMax); // any value is valid. No validation is required
    // horizontal offset, default = 0
    int xOffset = 0; // horizontal, row wise (positive right)
    if (argXOffset)
        xOffset = args::get(argXOffset); // any value is valid. No validation is required
    // vertical offset, default = 0
    int yOffset = 0; // vertical, column wise (positive down)
    if (argYOffset)
        yOffset = args::get(argYOffset); // any value is valid. No validation is required
    // output size, default same as input
    unsigned int xSize = 227; // width of output image (columns), 0 means use the same as input
    if (argXSize)
        xSize = args::get(argXSize); // any value is valid. No validation is required
    unsigned int ySize = 227;        // vertical, column wise (positive down)
    if (argYSize)
        ySize = args::get(argYSize); // any value is valid. No validation is required
    int argInteger = 0;
    if (argIntParam)
        argInteger = args::get(argIntParam); // any value is valid. No validation is required

    int bitsPerPixel = T2P_BPP8;
    if (argOutputBitDepth)
    {
        // check user defined number of bits per pixel
        int c = args::get(argOutputBitDepth);
        if ((c == T2P_BPP8) || (c == T2P_BPP16))
            bitsPerPixel = c;
        else
        {
            s << "Unknown/unsupported number of bits-per-pixel [" << yellow << c << reset << "]. Options are 8 & 16";
            logc.error("main:argOutputChannels", s);
            return -1;
        }
    }

    int outputChannels = T2P_GRAYSCALE; // default value for number of channel (1: Grayscale)
    if (argOutputChannels)
    {
        // check user defined value for argOutputChannels
        int c = args::get(argOutputChannels);
        if ((c == T2P_GRAYSCALE) || (c == T2P_RGB))
            outputChannels = c;
        else // unknown configuration for output channels. Print error message
        {
            s << "Unknown/unsupported number of channels [" << yellow << c << reset << "]. Options are 1 & 3";
            logc.error("main:argOutputChannels", s);
            return -1;
        }
    }
    // exported image size can be any positive value. if zero any of the dimensions, the it is assumed it will inherit the input image size for that dimension
    // potential silent bugs? maybe, if careless arg parsing is done during batch call from bash
    // minDepth < maxDepth
    // xOffset, yOffset may fall out-of-boudary. We check that after reading the input image

    /* Summary list parameters */
    if (verbosity >= 1)
    {
        cout << yellow << "****** Summary **********************************" << reset << endl;
        cout << "Input file:    \t" << yellow << inputFileName << reset << endl;
        cout << "Output file:   \t" << green << outputFileName << reset << endl;
        s << "Output file format. Channels: " << outputChannels << "\tBits per pixel: " << bitsPerPixel;
        logc.info("main", s);
        // TODO: implement export of transformed geoTIFF. (AGAIN? I think this was already implemented in previous releases)
        // if (argExportTiff)
        //     cout << "outputTIFF:    \t" << green << outputTIFF << reset << endl;; //extra geotiff copy to be exported
        cout << "validThreshold:\t" << yellow << validThreshold << reset << endl;
        cout << "ROI Offset:    \t(" << xOffset << ", " << yOffset << ")\tRotation: \t" << rotationAngle << "deg" << endl;
        if (xSize * ySize > 0)
            cout << "ROI Size:      \t(" << xSize << ", " << ySize << ")" << endl;
        else
            cout << "ROI Size: " << light_green << "<same as input>" << reset << endl;
    }

    // // Step 1: Read input TIFF file

    // create the container and the open input file
    Geotiff inputGeotiff(inputFileName.c_str());
    if (!inputGeotiff.isValid())
    { // check if nothing wrong happened with the constructor
        s << "Error opening geoTIFF file: " << yellow << inputFileName;
        logc.error("rl::readTIFF", s);
        return -1;
    }

    double transformMatrix[6];
    int layerDimensions[3];
    std::string layerProjection;

    //**************************************
    // Get/print summary information of the TIFF
    GDALDataset *poDataset;
    poDataset = inputGeotiff.GetDataset(); // pull the pointer to the main GDAL dataset structure
    // store a copy of the geo-transormation matrix
    poDataset->GetGeoTransform(transformMatrix);
    inputGeotiff.GetDimensions(layerDimensions);
    layerProjection = std::string(inputGeotiff.GetProjection());

    float **apData; // pull 2D float matrix containing the image data for Band 1
    apData = inputGeotiff.GetRasterBand(1);
    if (apData == nullptr)
    {
        s << "Error opening Geotiff file: " << yellow << inputFileName;
        logc.error("rl::readTIFF", "Error reading input geoTIFF data: NULL");
        return -1;
    }

    cv::Mat rasterData(layerDimensions[1], layerDimensions[0], CV_64FC1);
    cv::Mat tiff(layerDimensions[1], layerDimensions[0], CV_64FC1); // cv container for tiff data . WARNING: cv::Mat constructor is failing to initialize with apData
    // cout << "Dim: [" << layerDimensions[0] << "x" << layerDimensions[1] << endl;
    for (int i = 0; i < layerDimensions[1]; i++)
    {
        for (int j = 0; j < layerDimensions[0]; j++)
        {
            tiff.at<double>(cv::Point(j, i)) = (double)apData[i][j]; // swap row/cols from matrix to OpenCV container
        }
    }
    tiff.copyTo(rasterData);

    double noDataValue = inputGeotiff.GetNoDataValue();
    // updateMask();
    // updateStats();
    // return NO_ERROR;

    cv::Mat rasterMask = cv::Mat(rasterData.size(), CV_64FC1); // create global valid_data maskthat won't be updated
    cv::compare(rasterData, noDataValue, rasterMask, CMP_NE);  // ROI at the source data level

    //******************************************
    cv::Mat original(rasterData.size(), CV_64FC1);
    cv::Mat mask = rasterMask.clone();
    rasterData.copyTo(original, mask); // copy only valid pixels, the rest should remain zero
    // now, we need to extract ROI for the given rotation and displacement (offset)
    // 0/verify if sampling is necessary and validate sizes (update if not provided with input image params)
    if (xSize > mask.cols)
    {
        s << "Desired image output width larger than image input width (" << red << xSize << " > " << mask.cols << reset << ")";
        logc.error("validation", s);
        return -1;
    }
    else if (xSize == 0)
        xSize = mask.cols; // asked for autodetection of image width
    if (ySize > mask.rows)
    {
        s << "Desired image output height larger than image input height (" << red << ySize << " > " << mask.rows << reset << ")";
        logc.error("validation", s);
        return -1;
    }
    else if (ySize == 0)
        ySize = mask.rows; // asked for autodetection of image height

    // 1/compute the new center
    int cx = original.cols / 2;
    int cy = original.rows / 2; // center of the source image
    int nx = cx + xOffset;
    int ny = cy + yOffset; // to be used for the lat/lon or UTM coordinates and to determine the corners of the ROI defining the large bbox
    int nx_east = nx;
    int ny_north = ny;
    // 2/compute the max radius of the ROI (factoring rotation)
    int diag = ceil(sqrt(xSize * xSize + ySize * ySize) / 2.0) + 1; // radius of internal bounding box. The size of the external bbox is twice this radius
    // 2.1/verify crop extent still falls withint the src image
    int tlx = nx - diag; // top left corner
    int tly = ny - diag;
    int brx = nx + diag; // bottom right corner
    int bry = ny + diag;

    if (argIntParam)
    {
        if (verbosity >= 2)
            logc.warn("Override", "Using full image canvas for direct image export");
        tlx = 0;
        tly = 0;
        brx = original.cols - 1;
        bry = original.rows - 1;
    }
    // Validate ROI bounding boxes
    if (tlx < 0)
    {
        logc.error("rect", "top left corner X out of range (negative)");
        return -1;
    }
    if (tly < 0)
    {
        logc.error("rect", "top left corner Y out of range (negative)");
        return -1;
    }
    if (brx >= original.cols)
    {
        s << "bottom right corner X out of range: " << brx << " > " << original.cols;
        logc.error("rect", s);
        return -1;
    }
    if (bry >= original.rows)
    {
        s << "bottom right corner Y out of range: " << bry << " > " << original.rows;
        logc.error("rect", s);
        return -1;
    }

    cv::Mat final;
    if (argIntParam)
    { // argIntParam invoked, meaning we override rotation
        if (verbosity > 1)
            logc.warn("main", "Xfering input image to final container, no ROI cropping/extraction");

        // Check if source and destination image sizes are different. If so, crop the source image to size_x and size_y.
        if ((original.cols != xSize) || (original.rows != ySize))
        {
            tlx = floor((original.cols - xSize) / 2);
            tly = floor((original.rows - ySize) / 2);
            cv::Mat crop_roi(original, cv::Rect2d(tlx, tly, xSize, ySize)); // bbox of sie (xSize,ySize) centered at (tlx,tly)
            crop_roi.copyTo(final);
        }
        else
        {
            original.copyTo(final);
        }
    }
    else
    {
        // 3/crop large extent
        cv::Mat large_roi(original, cv::Rect2d(tlx, tly, 2 * diag, 2 * diag)); // the bbox size is twice the diagonal
        cv::Mat large_crop;
        large_roi.copyTo(large_crop);
        // 4/rotate given angle
        cv::Mat r = cv::getRotationMatrix2D(cv::Point2f((large_crop.cols - 1) / 2.0, (large_crop.rows - 1) / 2.0), rotationAngle, 1.0);
        // determine bounding rectangle, center not relevant
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), large_crop.size(), rotationAngle).boundingRect2f(); // this was a bit overkill
        // adjust transformation matrix
        r.at<double>(0, 2) += bbox.width / 2.0 - large_crop.cols / 2.0;
        r.at<double>(1, 2) += bbox.height / 2.0 - large_crop.rows / 2.0;
        cv::Mat rotatedROI;
        cv::warpAffine(large_crop, rotatedROI, r, bbox.size(), cv::INTER_NEAREST); // using nearest: faster and we avoid interpolation of nodata field
        // 5/crop small extent (xSize, ySize)
        tlx = rotatedROI.cols / 2 - xSize / 2; // center - width
        tly = rotatedROI.rows / 2 - ySize / 2; // center - height
        bbox = cv::Rect2d(tlx, tly, xSize, ySize);
        cv::Mat final_roi = rotatedROI(bbox); // crop the final size image, already rotated
        final_roi.copyTo(final);
    }

    // 6/update mask: compare against nodata field
    double nodata = noDataValue;
    // let's inspect the 'final' matrix and compare against  'nodata'
    cv::Mat final_mask;
    // nodata was replaced by ZERO when loading into the original matrix
    cv::compare(final, 0, final_mask, CMP_NE);
    // 7/normalize the mask. The actual PNG output must be scaled according to the bathymetry range param
    if (verbosity >= 2)
    {
        namedWindow("original");
        cv::normalize(original, original, 0, 255, NORM_MINMAX, CV_8UC1, mask); // normalize within the expected range 0-255 for imshow
        imshow("original", original);
        namedWindow("final_mask"); // already normalized
        imshow("final_mask", final_mask);
    }

    // 2.1) Compute the mean of the valid pixels. We need the number of valid pixels
    double _min, _max, _mean, _sum;
    int totalPixels = final.rows * final.cols;   // max total pixels, should be the same as xSize * ySize
    int totalValids = countNonZero(final_mask);  // total valid pixels = non zero pixels
    int totalZeroes = totalPixels - totalValids; // total invalid pxls = total - valids
    double proportion = (double)totalValids / (double)totalPixels;

    cv::Mat zero_mask = cv::Mat::zeros(final_mask.size(), CV_64FC1); // float zero mask
    zero_mask.copyTo(final, ~final_mask);                            // we copy zeros to the invalid points (negated mask). Cheaper to peform memory copy than multiplying by the mask
    _sum = cv::sum(final).val[0];                                    // checked in QGIS< ok. Sum all pixels including the invalid ones, which have been already converted to zero
    _mean = _sum / (float)totalValids;

    if (verbosity >= 2)
    {
        cv::minMaxLoc(final, &_min, &_max, 0, 0, final_mask); // masked min max of the input bathymetry
        cout << light_yellow << "RAW bathymetry - \t" << reset << "MIN / MEAN / MAX = [" << _min << " / " << _mean << " / " << _max << "]" << endl;
    }
    // 2.2) Shift the whole map to the mean (=0)
    if (argInteger != COPY_ONLY_MODE)
    {
        cv::subtract(final, _mean, final, final_mask); // MEAN centering of only valida data points (use mask)
        if (verbosity >= 2)                            // print mean substraction message
        {
            cout << light_purple << "Substracting mean value from bathymetry: " << reset << _mean << endl;
        }
    }
    // show debug
    if (verbosity >= 2)
    {
        // cv::normalize(final, final, 0, 255, NORM_MINMAX, CV_8UC1, final_mask); // normalize within the expected range 0-255 for imshow
        namedWindow("final");
        cv::Mat temp;
        cv::normalize(final, temp, 0, 255, NORM_MINMAX, CV_8UC1, final_mask); // normalize within the expected range 0-255 for imshow
        imshow("final", temp);
        // recompute min/max
        cv::minMaxLoc(final, &_min, &_max, 0, 0, final_mask); // masked min max of the input bathymetry
        _sum = cv::sum(final).val[0];                         // checked in QGIS< ok. Sum all pixels including the invalid ones, which have been already converted to zero
        _mean = _sum / (float)totalValids;
        cout << light_green << "Adjusted bathymetry - \t" << reset << "MIN / MEAN / MAX = [" << _min << " / " << _mean << " / " << _max << "]" << endl;
        waitKey(0);
    }
    // duplicate for export
    cv::Mat final_png;
    final.copyTo(final_png); // copy for normalization to 0-255. The source can be used to be exported as local bathymetry geoTIFF
    // 2.4) Scale to 255/2 (8 bits) or 65535/2 for 16 bits
    double max_range = (1 << bitsPerPixel) / 2.0;
    double offset = max_range;
    if (argInteger != COPY_ONLY_MODE)
    {
        final_png = final_png * (double)(max_range / maxDepth); // Rescale terrain to fit within image format. maxDepth will match max_range
        final_png = final_png + max_range;                      // 1-bit bias. The new ZERO should be in the center of the range
        cout << light_green << "Rescaling bathymetry to fit within " << bitsPerPixel << " bits: " << reset << max_range << endl;
    }
    if (verbosity >= 2)
    {
        double png_mean = (double)cv::sum(final_png).val[0] / (double)(final_png.cols * final_png.rows);
        cv::minMaxLoc(final_png, &_min, &_max, 0, 0, final_mask); // debug
        // fancy colour to indicate if out of range [0, 255]. his is a symptom of depth range saturation for the lcoal bathymetry patch
        cout << light_blue << "Exported PNG image - \t" << reset << "MIN / MEAN / MAX = [" << ((_min < 0.0) ? red : green) << _min << reset << " / " << png_mean;
        cout << " / " << ((_min > 255.0) ? red : green) << _max << reset << "]" << endl;
    }
    // Step 3: use geoTransform matrix to retrieve center of map image
    // coordinates are given as North-East positive. Vertical resolution sy (coeff[5]) can be negative
    // as long as the whole dataset is self-consistent, analysis based on relative distance will ignore any offset (e.g. neighbour tables)
    // Before exporting the geoTIFF, we need to correct the geotransformation matrix to reflect the x/y offset
    // it should be mapped as a northing/easting displacement (scaled by the resolution)
    // easting_offset -> transforMatrix[0]
    // northing_offset -> transforMatrix[3]

    // // this is the northing/easting in the original reference system
    double easting = transformMatrix[0] + transformMatrix[1] * nx_east;   // easting
    double northing = transformMatrix[3] + transformMatrix[5] * ny_north; // northing
    // // easily, we can add those projected (UTM) coordinates as the new offset (careful: center ref vs corner ref)
    transformMatrix[0] = easting - (final.cols / 2) * transformMatrix[1];
    transformMatrix[3] = northing - (final.rows / 2) * transformMatrix[5];

    // Feature: do not export images with a proportion of valid pixels lower than a threshold
    if (proportion >= validThreshold)
    { // Set threshold to 0.0 to export all images
        // before exporting, we check the desired number of image channels and bits per pixel
        if (bitsPerPixel == T2P_BPP8)
        {
            final_png.convertTo(final_png, CV_8UC1);
        }
        else
        {
            final_png.convertTo(final_png, CV_16UC1);
        }
        // check if image needs to be converted to 3-channel (RGB-like) format
        if (outputChannels == T2P_RGB)
        { // we need to convert to RGB
            // final_png.convertTo(final_png, CV_16UC3);
            cv::cvtColor(final_png, final_png, COLOR_GRAY2RGB);
        }
        cv::imwrite(outputFileName, final_png);
    }
    // ---------------------------------------------------------------------------------------
    // Exporting georef information to the console so it can be collected by caller scripts
    // We need the LAT LON in decimal degree to match oplab-pipeline and LGA input format
    double latitude;
    double longitude;
    // Transform from northing easting to WGS84 lat lon
    OGRSpatialReference refUtm;
    refUtm.importFromProj4(layerProjection.c_str()); // original SRS from source image
    OGRSpatialReference refGeo;
    // If asked for, we use Mars CRS
    if (argCRS)
    { // switch to Mars Lat/Lon CRS IAU2000:49001
        if (verbosity > 0)
            cout << light_yellow << "Using PROJ4 CRS definition for Mars as celestial body" << endl;
        // +proj=longlat +a=3396190 +rf=169.894447223612 +no_defs +type=crs
        // const char *crsIAU2000_49900 = "GEOGCRS["Mars 2000",
        // DATUM["D_Mars_2000",
        //     ELLIPSOID["Mars_2000_IAU_IAG",3396190,169.894447223612,
        //         LENGTHUNIT["metre",1,
        //             ID["EPSG",9001]]]],
        // PRIMEM["Greenwich",0,
        //     ANGLEUNIT["Decimal_Degree",0.0174532925199433]],
        // CS[ellipsoidal,2],
        //     AXIS["longitude",east,
        //         ORDER[1],
        //         ANGLEUNIT["Decimal_Degree",0.0174532925199433]],
        //     AXIS["latitude",north,
        //         ORDER[2],
        //         ANGLEUNIT["Decimal_Degree",0.0174532925199433]]]

        //         ";
        refGeo.importFromProj4("+proj=longlat +a=3396190 +rf=169.894447223612 +no_defs +type=crs");
    }
    else
    {
        refGeo.SetWellKnownGeogCS("WGS84"); // target CRS: Earth - default
    }

    OGRCoordinateTransformation *coordTrans = OGRCreateCoordinateTransformation(&refUtm, &refGeo); // ask for a SRS transforming object
    double x = easting;
    double y = northing;

    cout.precision(std::numeric_limits<double>::digits10); // maybe it's an overkill. Just in case
    int reprojected = coordTrans->Transform(1, &x, &y);
    latitude = y; // yes, this is not a bug, they are swapped
    longitude = x;
    delete coordTrans; // maybe can be removed as destructor and garbage collector will take care of this after return
    String separator = "\t";
    if (argCsv)
        separator = ",";

    if (verbosity >= 1)
    {
        // export header colums
        cout << "valid_ratio" << separator;
        // cout << "relative_path"     << separator; // this information is know by the caller, no need to export
        cout << "northing [m]" << separator;
        cout << "easting [m]" << separator;
        cout << "depth [m]" << separator; // it could be 'altitude' for topography maps
        cout << "latitude [deg]" << separator;
        cout << "longitude [deg]" << endl;
    }
    // export data columns (always)
    cout << proportion << separator; // proportion of valid pixels, can be used by the caller to postprocessing culling
    cout << northing << separator;   // northing [m]
    cout << easting << separator;    // easting [m]
    cout << _mean << separator;      // mean depth for the current bathymety patch
    cout << latitude << separator;   // mean depth for the current bathymety patch
    cout << longitude << separator;  // mean depth for the current bathymety patch
    cout << endl;
    return NO_ERROR;
}
