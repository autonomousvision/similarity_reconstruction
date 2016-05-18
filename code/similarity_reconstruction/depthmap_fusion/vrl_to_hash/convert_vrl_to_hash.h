/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
/** converts the variable run length representation used in the original Volumetric fusion code
 * to hash representation
 * Original representation defined in OccGridRLE.h
 */
#include <string>

class OccGridRLE;
namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf
{
bool ConvertRLEGridToHashMap(OccGridRLE& rle_grid, int original_ramp_size, cpu_tsdf::TSDFHashing* hash_grid, bool init_hash);
bool ReadFromVRIFile(const std::string& filepath, int original_ramp_size, cpu_tsdf::TSDFHashing* hash_grid, bool init_hash = true);
}
