/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <Eigen/Eigen>
namespace cpu_tsdf {
class OrientedBoundingBox;
}
namespace tsdf_utility {
class OrientedBoundingBox;
}
namespace tsdf_test {
// test intersection only on xy plane
bool TestOBBsIntersection2D(
        const tsdf_utility::OrientedBoundingBox& obb1,
        const tsdf_utility::OrientedBoundingBox& obb2
        );

double OBBIntersectionVolume(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2);

double OBBIOU(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2);

bool OBBSimilarity(
        const tsdf_utility::OrientedBoundingBox& obb1,
        const tsdf_utility::OrientedBoundingBox& obb2,
        float* similarity
        );

bool TestOBBIntersection3D(
        const tsdf_utility::OrientedBoundingBox& a,
        const tsdf_utility::OrientedBoundingBox& b
        );

bool TestOBBsIntersection3D(
        const tsdf_utility::OrientedBoundingBox& obb,
        const std::vector<tsdf_utility::OrientedBoundingBox>& test_obbs);

double unitOverlap(const Eigen::Matrix4f& Tr);

double UnitOverlap2D(const Eigen::Matrix3f& Tr);

}  // end namespace
