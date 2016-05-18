/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <string>
#include <iostream>

#include <Eigen/Eigen>
#include <pcl/PolygonMesh.h>

// cpu_tsdf::OrientedBoudingBox is the deprecated class for OBB.
// use tsdf_utility::OrientedBoundingBox instead
namespace cpu_tsdf {
class OrientedBoundingBox;
}

namespace tsdf_utility {
// the oriented bounding box (OBB) of samples, represented by an affine transform
// transforming a unit cube at  [-0.5,0.5] for x and y and [0,1] for z to the detection OBB.
class OrientedBoundingBox {
public:
    friend std::ostream& operator << (std::ostream& ofs, const OrientedBoundingBox& obb);
    friend std::istream& operator >> (std::istream& ifs, OrientedBoundingBox& obb);
    OrientedBoundingBox();
    explicit OrientedBoundingBox(const Eigen::AffineCompact3f& transform)
            : transform_(transform) {}
   // Composing OBB from side lengths, center position and rotation angle around z axis
   // no limitation on the range of angle
   OrientedBoundingBox(const Eigen::Vector3f& side_lengths, const Eigen::Vector3f& bottom_center, const double angle);
   OrientedBoundingBox(const Eigen::Vector3f& side_lengths, float cx, float cy, float cbz, const double angle);
   int WriteToPly(const std::string& filename) const;
   int WriteToFile(const std::string& filename) const;
   OrientedBoundingBox ExtendSides(const Eigen::Vector3f& extension) const;
   OrientedBoundingBox ExtendSidesByPercent(const Eigen::Vector3f& extension_percent) const;
   double Volume() const { return SideLengths().prod(); }
   double AngleRangeTwoPI() const;
   double AngleRangePosNegPI() const;
   inline const Eigen::Vector3f OBBPos() const { return Eigen::Vector3f(transform_.translation()[0], transform_.translation()[1], AngleRangeTwoPI()); }
   const Eigen::Affine2f Affine2D() const;  // not considering z-axis translation
   const Eigen::Affine3f AffineTransform() const;
   const Eigen::Affine3f AffineTransformOriginAsCenter() const;
   inline Eigen::Vector3f BottomCenter() const { return transform_.translation(); }
   inline Eigen::Vector3f Center() const { return transform_ * Eigen::Vector3f(0, 0, 0.5); }
   inline Eigen::Vector3f Offset() const { return transform_ * Eigen::Vector3f(-0.5, -0.5, 0); }
   inline Eigen::Vector3f SideLengths() const { return transform_.linear().colwise().norm(); }
   inline Eigen::Matrix3f Orientations() const { return transform_.linear() * SideLengths().cwiseInverse().asDiagonal(); }
   inline Eigen::Vector3f SamplingDeltas(const Eigen::Vector3i& sample_size) const {
       return SideLengths().cwiseQuotient((sample_size - Eigen::Vector3i(1, 1, 1)).cast<float>());
   }
   inline Eigen::Matrix3f SamplingOrientedDeltas( const Eigen::Vector3i& sample_size ) const {
       return transform_.linear() * (sample_size - Eigen::Vector3i(1, 1, 1)).cast<float>().cwiseInverse().asDiagonal();
   }
   const Eigen::AffineCompact3f& transform() const { return transform_; }
   void transform(const Eigen::AffineCompact3f& t) { transform_ = t; }
private:
   void ComputeVertices(Eigen::Matrix<float, 3, 8>* obb_vertices) const;
   void ToPolygonMesh(pcl::PolygonMesh* pmesh) const;
   Eigen::AffineCompact3f transform_;
};
std::istream& operator >> (std::istream& ifs, OrientedBoundingBox& obb);
std::ostream& operator << (std::ostream& ofs, const OrientedBoundingBox& obb);
void OutputOBBsAsPly(const std::vector<OrientedBoundingBox>& obbs, const std::string & filename, std::vector<std::string> *saved_filelist = NULL);
void OutputAnnotatedOBB(const std::vector<std::vector<tsdf_utility::OrientedBoundingBox> > &obbs, const std::string &filename);
void OutputAnnotatedOBB(const std::vector<OrientedBoundingBox>& obb_vec, const std::vector<int>& sample_model_idx, const std::string& filename);
void InputAnnotatedOBB(const std::string& filename, std::vector<std::vector<tsdf_utility::OrientedBoundingBox>>* obbs);
cpu_tsdf::OrientedBoundingBox OldOBBFromNew(const tsdf_utility::OrientedBoundingBox& obb);
std::vector<cpu_tsdf::OrientedBoundingBox> OldOBBsFromNew(const std::vector<tsdf_utility::OrientedBoundingBox>& obbs);
OrientedBoundingBox NewOBBFromOld(const cpu_tsdf::OrientedBoundingBox& old_obb);
std::vector<OrientedBoundingBox> NewOBBsFromOlds(const std::vector<cpu_tsdf::OrientedBoundingBox>& old_obbs);
}  // namespace tsdf_utility
