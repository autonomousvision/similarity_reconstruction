/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
/**
 * Diffusion hole filling
 * Iteratively smooth the TSDF to fill holes in the TSDF representation.
 */
namespace cpu_tsdf
{
class TSDFHashing;
}

namespace cpu_tsdf
{
/**
 * Diffusion hole filling
 * Iteratively smooth the TSDF to fill holes in the TSDF representation.
 * @param tsdf_volume: the input TSDF, it's modified during the process.
 * @param iterations: how many iterations are executed.
 * @param new_tsdf_volume: the output TSDF.
 */
void DiffusionHoleFilling(TSDFHashing* tsdf_volume, int iterations, TSDFHashing* new_tsdf_volume);
}
