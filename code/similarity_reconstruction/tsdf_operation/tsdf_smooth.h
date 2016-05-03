/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
/**
 *  Functions for smoothing a TSDF.
 */
//#include <opencv2/opencv.hpp>
namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf
{
/**
* @brief Smooth the TSDF.
* @param tsdf_volume the tsdf to be smoothed
* @param neighborhood the neighborhood used for smoothing. A cube of size (2 * neighborhood + 1)^3 is used.
* usually set to 1
* @param new_tsdf_volume the smoothed tsdf. (cannot be the same as tsdf_volume)
*/
void SmoothTSDFFunction(const TSDFHashing* tsdf_volume, int neighborhood, TSDFHashing* new_tsdf_volume);
/** Smooth the TSDF but not modifying the TSDF points originally with values. (i.e. only "dilation")
*/
void SmoothTSDFFunctionOnlyExtension(const TSDFHashing* tsdf_volume, int neighborhood, TSDFHashing* new_tsdf_volume);
/** Smooth the TSDF but not modifying the TSDF points originally with values. (i.e. only "dilation")
 *  With a Gaussian kernel.
*/
void SmoothTSDFFunctionOnlyExtensionGaussian(const TSDFHashing* tsdf_volume, int neighborhood,
                                             double sigma, TSDFHashing* new_tsdf_volume);
}
