/**
 * @file options.h
 * @brief Argument parser options based on args.hxx. Extended to accomodate multiple modules using similar parsers
 * @version 1.1
 * @date 18/06/2020
 * @author Jose Cappelletto
 */

#ifndef _PROJECT_OPTIONS_H_

#define _PROJECT_OPTIONS_H_

#define ERROR_WRONG_ARGUMENT    -1
#define ERROR_MISSING_ARGUMENT  -2 
#include "headers.h"
#include "../external/args.hxx"
#include <iostream>

args::ArgumentParser argParser("","");
args::HelpFlag 	     argHelp(argParser, "help", "Display this help menu", {'h', "help"});
args::CompletionFlag completion(argParser, {"complete"});	//TODO: figure out why is missing in current version of args.hxx

args::ValueFlag <std::string> 	argInput(argParser,     "input", "Input geoTIFF image that contains the terrain information", {'i', "input"});
args::ValueFlag	<std::string> 	argOutput(argParser,    "filename", "Output PNG file",                          {'o', "output"});
args::ValueFlag	<int> 	        argVerbose(argParser,   "verbose",  "Define verbosity level [0-2]",               {'v', "verbose"});
// args::ValueFlag	<std::string> 	argExportTiff(argParser,"filename", "GeoTIFF copy of the exported image (experimental)",   {'e', "export_tiff"});

// File output parameters: number of channels (1-grayscale, 3-RGB) & pixel depth (8,16 bits)
args::ValueFlag <int>           argOutputChannels(argParser, "1|3", "Output image channels. 1: Grayscale (default), 3: RGB", {'c', "channels"});
args::ValueFlag <int>           argOutputBitDepth(argParser, "8|16", "Bits per pixel. All images are integer: 8 (default), 16", {'b', "bits"});

// Free parameters for debugging
args::ValueFlag	<int> 	argIntParam(argParser,  "param",    "Enable/disable full canvas export. When enabled it will override any ROI settings",  {"int"});
args::ValueFlag	<float> argFloatParam(argParser,"param",    "User defined parameter FLOAT for testing purposes",    {"float"});
// Sampling parameters
args::ValueFlag	<double>        argRotation(argParser,"degrees",  "Rotation angle of the ROI to be exported [degrees]",   {"rotation"});
args::ValueFlag	<int>           argXOffset(argParser,"pixels", "ROI horizontal (X) offset from the input image center", {"offset_x"});
args::ValueFlag	<int>           argYOffset(argParser,"pixels", "ROI vertical (Y) offset from the input image center",   {"offset_y"});
args::ValueFlag	<unsigned int>  argXSize(argParser,"pixels", "ROI width (X) in pixels",                                 {"size_x"});
args::ValueFlag	<unsigned int>  argYSize(argParser,"pixels", "ROI height (Y) in pixels",                                {"size_y"});
args::ValueFlag	<double>        argZMax(argParser,"meters", "Maximum input value (Z). It wil be mapped to 255",         {"max_z"});
// Thresholds
args::ValueFlag	<double>        argValidThreshold(argParser,"ratio", "Minimum ratio of required valid pixels to generate PNG",{"valid_th"});
args::Flag	         	        argCsv(argParser,   "",  "Use comma ',' as column separator rather than TAB",           {"csv"});
// CRS and geoTIFF related flags
args::Flag	         	        argCRS(argParser,  "",  "Switch Lat/Lon CRS from WGS84 (Earth) projection to IAU2000:49901 (Mars)", {"crs"});

/**
 * @brief Inititalize argument parser for tiff2png module
 * 
 * @param argc cli argc (count)
 * @param argv cli argv (values)
 * @param newDescription User-defined module description
 * @return int error code if any
 */
int initParser(int argc, char *argv[], string newDescription = ""){
    /* PARSER section */
    std::string descriptionString =
        "tiff2png - image preprocessing tool for LGA/geoCLR + BNN based terrain inference engine \
        Partial data augmentation on demand by resampling input image, via traslation and rotation \
        Data range linear remapping with (clip-limit) is performed beore exporting as PNG image \
    Compatible interface with geoTIFF bathymetry datasets via GDAL + OpenCV";

    if (!newDescription.empty())
        argParser.Description(newDescription);
    else
        argParser.Description(descriptionString);
    
    argParser.Epilog("Author: J. Cappelletto (GitHub: @cappelletto)\n");
    argParser.Prog(argv[0]);
    argParser.helpParams.width = 120;

    try
    {
        argParser.ParseCLI(argc, argv);
    }
    catch (const args::Completion &e)
    {
        cout << e.what();
        return 0;
    }

    catch (args::Help)
    { // if argument asking for help, show this message
        cout << argParser;
        return ERROR_MISSING_ARGUMENT;
    }
    catch (args::ParseError e)
    { //if some error ocurr while parsing, show summary
        std::cerr << e.what() << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return ERROR_WRONG_ARGUMENT;
    }
    catch (args::ValidationError e)
    { // if some error at argument validation, show
        std::cerr << "Bad input commands" << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return ERROR_WRONG_ARGUMENT;
    }
    return 0;
}



#endif //_PROJECT_OPTIONS_H_