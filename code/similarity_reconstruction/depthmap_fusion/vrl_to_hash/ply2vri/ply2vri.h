#pragma once
#include <string>

class OccGridRLE;
struct PLY2VRIOptions {
    int rampSize;
    int exact_ramp_size;
    double resolution;
    bool hardEdges;
    bool useConf ;
    int stripEdges;
    int algorithm;
    OccGridRLE *templateVRI;
    std::string input_filename;
    std::string output_filename;
};
int ReadPLYAndConvertToVRI(const PLY2VRIOptions& options);
