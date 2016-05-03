/*
 * Marching cubes for TSDF representation, core part from PCL library.
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/impl/marching_cubes.hpp>

#include "../tsdf_representation/tsdf_hash.h"

namespace cpu_tsdf
{

pcl::PolygonMesh::Ptr TSDFToPolygonMesh(TSDFHashing::ConstPtr tsdf_model, float mesh_min_weight, float flatten_dist = -1);

  class MarchingCubesTSDFHashing: public pcl::MarchingCubes<pcl::PointXYZ>
  {
  public:
    MarchingCubesTSDFHashing():
      w_min_ (0),
      pcl::MarchingCubes<pcl::PointXYZ> ()
    {}
    virtual ~MarchingCubesTSDFHashing(){};

    void
    setInputTSDF (cpu_tsdf::TSDFHashing::ConstPtr tsdf_volume);

    bool
    getValidNeighborList1D (std::vector<float> &leaf,
                            Eigen::Vector3i &index3d);

    inline void
    setMinWeight (float w_min)
    { w_min_ = w_min;}
  protected:
///////////////////////////////////////
    inline bool
    getGridValueWithColor (const Eigen::Vector3i& pos, float* d, float* w, cv::Vec3b* color)
    {
        return ((tsdf_volume_->RetriveData(pos, d, w, color)) && *w > w_min_);
    }

    bool
    getValidNeighborList1DWithColor (const Eigen::Vector3i &index3d,
            std::vector<float> &leaf,
            std::vector<cv::Vec3b> &color_leaf);

    inline void
    interpolateColor (const cv::Vec3b &p1,
            const cv::Vec3b &p2,
            float val_p1,
            float val_p2,
            cv::Vec3b &output)
    {
        float mu = (iso_level_ - val_p1) / (val_p2-val_p1);
        output = static_cast<cv::Vec3b>(p1 + mu * (p2 - p1));
    }

    template <typename PointNT> void
    createSurfaceWithColor (std::vector<float> &leaf_node,
            const std::vector<cv::Vec3b>& leaf_color,
            Eigen::Vector3i &index_3d,
            pcl::PointCloud<PointNT> &cloud);

    void
    reconstructVoxelWithColor (pcl::PointCloud<pcl::PointXYZRGB> &output);
/////////////////////////////////////
    void
    voxelizeData ();

    float
    getGridValue (Eigen::Vector3i pos);

    void
    performReconstruction (pcl::PolygonMesh &output);

    void
    reconstructVoxel (pcl::PointCloud<pcl::PointXYZ> &output);

    cpu_tsdf::TSDFHashing::ConstPtr tsdf_volume_;
    float w_min_;
  };
}
