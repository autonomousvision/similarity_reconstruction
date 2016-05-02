#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cassert>
#include <functional>
#include <vector>
#include <string>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <boost/format.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "tsdf_feature_generate.h"
#include "tsdf_slice.h"
#include "utility/utility.h"
#include "common/utility/pcl_utility.h"
#include "common/utility/common_utility.h"
#include "boost/threadpool.hpp"
#include "tsdf_operation/tsdf_io.h"

#include "detection/detection_utility.h"
#include "detection/detect_sample.h"
#include "detection/detector.h"
#include "common/utility/timer.h"

//void cpu_tsdf::ComputeFeature(const Eigen::Matrix3f &bb_orientation, const Eigen::Vector3f &bb_sidelengths, const Eigen::Vector3f &bb_offset,
//                              const Eigen::Vector3f &voxel_lengths, const cpu_tsdf::TSDFHashing& tsdf_origin, cpu_tsdf::TSDFFeature *feature)
//{
//    Eigen::Vector3i voxel_size = (utility::ceil(
//                                      Eigen::Vector3f(bb_sidelengths.cwiseQuotient(voxel_lengths))
//                                      )).cast<int>();
//    int total_voxel_size = voxel_size[0] * voxel_size[1] * voxel_size[2];
//    feature->Clear();
//    feature->bb_sidelengths = bb_sidelengths;
//    feature->voxel_lengths = voxel_lengths;
//    feature->occupied_.resize(total_voxel_size);
//    Eigen::Matrix3f bb_orientation_scaled = bb_orientation * voxel_lengths.asDiagonal();
//    int cur_cnt = 0;
//    for (int ix = 0; ix < voxel_size[0]; ++ix)
//        for (int iy = 0; iy < voxel_size[1]; ++iy)
//            for (int iz = 0; iz < voxel_size[2]; ++iz)
//            {
//                Eigen::Vector3f cur_world_coord = bb_offset + (bb_orientation_scaled.col(0) * ix + bb_orientation_scaled.col(1) * iy + bb_orientation_scaled.col(2) * iz);
//                float cur_d = 0;
//                float cur_w = 0;
//                if (tsdf_origin.RetriveDataFromWorldCoord(cur_world_coord, &cur_d, &cur_w))
//                {
//                    feature->occupied_[cur_cnt] = true;
//                    feature->tsdf_vals.push_back(cur_d);
//                    feature->tsdf_weights.push_back(cur_w);
//                }
//                else
//                {
//                    feature->occupied_[cur_cnt] = false;
//                    feature->tsdf_vals.push_back(1);
//                    feature->tsdf_weights.push_back(0);
//                }
//                cur_cnt ++;
//            }
//}

//void cpu_tsdf::ComputeFeature(const Eigen::Matrix3f &bb_orientation, const Eigen::Vector3f &bb_sidelengths, const Eigen::Vector3f &bb_offset,
//                              const Eigen::Vector3i& bb_size, const cpu_tsdf::TSDFHashing& tsdf_origin, cpu_tsdf::TSDFFeature *feature)
//{
//    Eigen::Vector3f voxel_lengths = bb_sidelengths.cwiseQuotient(bb_size.cast<float>());
//    return ComputeFeature(bb_orientation, bb_sidelengths, bb_offset, voxel_lengths, tsdf_origin, feature);
//}

bool cpu_tsdf::GenerateOneSample(
        const cpu_tsdf::TSDFHashing &tsdf_origin,
        const cpu_tsdf::OrientedBoundingBox& bounding_box,
        const float template_occupied_ratio,
        const float mesh_min_weight,
        cpu_tsdf::TSDFFeature *out_feature)
{
    const cpu_tsdf::OrientedBoundingBox& cur_bb = bounding_box;
    out_feature->ComputeFeature(cur_bb, tsdf_origin);
    float cur_occupied_ratio = out_feature->occupied_ratio(mesh_min_weight);
    if (cur_occupied_ratio < template_occupied_ratio * 0.3)
    {
        out_feature->Clear();
        return false;
    }
    return true;
}

void cpu_tsdf::GenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
                               const std::vector<cpu_tsdf::OrientedBoundingBox> &bounding_boxes,
                               std::vector<cpu_tsdf::TSDFFeature> *features, std::vector<cpu_tsdf::TSDFHashing::Ptr> *feature_tsdfs)
{
    features->clear();
    feature_tsdfs->clear();
    for (int i = 0; i < bounding_boxes.size(); ++i)
    {
        const cpu_tsdf::OrientedBoundingBox& cur_bb = bounding_boxes[i];
        TSDFFeature cur_feat;
        cur_feat.ComputeFeature(cur_bb, tsdf_origin);

        if (feature_tsdfs)
        {
            TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
            SliceTSDF(&tsdf_origin, PointInOrientedBox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths),  cur_feat_tsdf.get());
            feature_tsdfs->push_back(cur_feat_tsdf);
        }
        features->push_back(cur_feat);
        if (i > 0)
        {
           CHECK_EQ(cur_feat.VoxelSideLengths()[0], features->back().VoxelSideLengths()[0]);
           CHECK_EQ(cur_feat.VoxelSideLengths()[1], features->back().VoxelSideLengths()[1]);
           CHECK_EQ(cur_feat.VoxelSideLengths()[2], features->back().VoxelSideLengths()[2]);
        }
    }
}

bool TestOBBsCompletelyOverlap(const cpu_tsdf::OrientedBoundingBox& obb, const std::vector<cpu_tsdf::OrientedBoundingBox>& test_obbs)
{
    static const float thresh = 1e-2;
    for (int i = 0; i < test_obbs.size(); ++i)
    {
        if (
                (obb.bb_offset - test_obbs[i].bb_offset).cwiseAbs().maxCoeff() < thresh &&
                (obb.bb_orientation - test_obbs[i].bb_orientation).cwiseAbs().maxCoeff() < thresh &&
                (obb.bb_sidelengths - test_obbs[i].bb_sidelengths).cwiseAbs().maxCoeff() < thresh
           )
        {
            return true;
        }
    }
    return false;
}


//check if two oriented bounding boxes overlap
#define VECTOR Eigen::Vector3f
#define SCALAR float
/*
 *  a: half sidelength
 * Pa: center
*/
bool OBBOverlap
(        //A
        VECTOR&	a,	//extents
        VECTOR&	Pa,	//position
        VECTOR*	A,	//orthonormal basis
        //B
        VECTOR&	b,	//extents
        VECTOR&	Pb,	//position
        VECTOR*	B	//orthonormal basis
);
bool TestOBBIntersection(const cpu_tsdf::OrientedBoundingBox& a,
                         const cpu_tsdf::OrientedBoundingBox& b);
bool TestOBBsIntersection(const cpu_tsdf::OrientedBoundingBox& obb, const std::vector<cpu_tsdf::OrientedBoundingBox>& test_obbs)
{
    for (int i = 0; i < test_obbs.size(); ++i)
    {
        if (TestOBBIntersection(obb, test_obbs[i]))
            return true;
    }
    return false;
}

bool TestOBBIntersection(const cpu_tsdf::OrientedBoundingBox& a,
                         const cpu_tsdf::OrientedBoundingBox& b)
{
    Eigen::Vector3f extent_a = a.bb_sidelengths/2.0;
    Eigen::Vector3f pos_a = a.bb_offset + a.bb_orientation * extent_a;
    Eigen::Vector3f basis_a[3] = {a.bb_orientation.col(0), a.bb_orientation.col(1), a.bb_orientation.col(2)};

    Eigen::Vector3f extent_b = b.bb_sidelengths/2.0;
    Eigen::Vector3f pos_b = b.bb_offset + b.bb_orientation * extent_b;
    Eigen::Vector3f basis_b[3] = {b.bb_orientation.col(0), b.bb_orientation.col(1), b.bb_orientation.col(2)};

    return OBBOverlap(extent_a, pos_a, basis_a,
                      extent_b, pos_b, basis_b);
}

/*not tested yet*/
double unitOverlap(const Eigen::Matrix4f& Tr);
//double OBBOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const cpu_tsdf::OrientedBoundingBox& obb2);
bool OBBsLargeOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const std::vector<cpu_tsdf::OrientedBoundingBox>& obbs,
                          int* intersected_box, float* intersect_area, const float thresh)
{
    float fintersect_area = 0;
    for (int i = 0; i < obbs.size(); ++i)
    {
        float cur_area = OBBOverlapArea(obb1, obbs[i]) ;
        if (cur_area > thresh)
        {
            if (cur_area > fintersect_area)
            {
                fintersect_area = cur_area;
                if (intersected_box) *intersected_box = i;
            }
        }
    }
    if (fintersect_area > 0)
    {
        if (intersect_area) *intersect_area = fintersect_area;
        return true;
    }
    else
    {
        if (intersect_area) *intersect_area = 0;
        if (intersected_box) *intersected_box = -1;
        return false;
    }

}

double OBBOverlapArea(const cpu_tsdf::OrientedBoundingBox& obb1, const cpu_tsdf::OrientedBoundingBox& obb2)
{
    if (!TestOBBIntersection(obb1, obb2)) return 0;
    Eigen::Matrix4f Tr1, Tr2;
    Tr1.setZero();
    Tr1.block(0, 0, 3, 3) = obb1.bb_orientation * obb1.bb_sidelengths.asDiagonal();
    Tr1.block(0, 3, 3, 1) = obb1.BoxCenter();
    Tr1(3, 3) = 1;

    Tr2.setZero();
    Tr2.block(0, 0, 3, 3) = obb2.bb_orientation * obb2.bb_sidelengths.asDiagonal();
    Tr2.block(0, 3, 3, 1) = obb2.BoxCenter();
    Tr2(3, 3) = 1;

    return (unitOverlap(Tr1.inverse() * Tr2) + unitOverlap(Tr2.inverse() * Tr1))/2.0;
}

double unitOverlap(const Eigen::Matrix4f& Tr)
{
    static const int sample_num = 21;
    static const int total_num = sample_num * sample_num * sample_num;
    static std::vector<Eigen::Vector4f> sample_pts;
    if (sample_pts.size() != total_num)
    {
        sample_pts.resize(total_num);
        int cnt = -1;
        for (int x = 0; x < sample_num; ++x)
            for (int y = 0; y < sample_num; ++y)
                for (int z = 0; z < sample_num; ++z)
                {
                    cnt++;
                    sample_pts[cnt] = (Eigen::Vector4f(x, y, z, 0) - Eigen::Vector4f(sample_num/2, sample_num/2, sample_num/2, 0)) / double((sample_num-1));
                    sample_pts[cnt](3) = 1;
                }
    }
    int overlap_cnt = 0;
    for (int i = 0; i < sample_pts.size(); ++i)
    {
        Eigen::Vector4f proj_pt = Tr * sample_pts[i];
        Eigen::Vector3f normed_pt = Eigen::Vector3f(proj_pt[0]/proj_pt[3], proj_pt[1]/proj_pt[3], proj_pt[2]/proj_pt[3]);
        if (fabs(normed_pt[0]) <= 0.5 && fabs(normed_pt[1]) <= 0.5 && fabs(normed_pt[2]) <= 0.5) overlap_cnt++;
    }
    return (double)overlap_cnt/total_num;
}


//check if two oriented bounding boxes overlap
/*
 *  a: half sidelength
 * Pa: center
*/
bool OBBOverlap
(        //A
        VECTOR&	a,	//extents
        VECTOR&	Pa,	//position
        VECTOR*	A,	//orthonormal basis
        //B
        VECTOR&	b,	//extents
        VECTOR&	Pb,	//position
        VECTOR*	B	//orthonormal basis
)
{
    //translation, in parent frame
    VECTOR v = Pb - Pa;
    //translation, in A's frame
    VECTOR T( v.dot(A[0]), v.dot(A[1]), v.dot(A[2]) );

    //B's basis with respect to A's local frame
    SCALAR R[3][3];
    float ra, rb, t;
    long i, k;

    //calculate rotation matrix
    for( i=0 ; i<3 ; i++ )
        for( k=0 ; k<3 ; k++ )
            R[i][k] = A[i].dot(B[k]);
    /*ALGORITHM: Use the separating axis test for all 15 potential
separating axes. If a separating axis could not be found, the two
boxes overlap. */

    //A's basis vectors
    for( i=0 ; i<3 ; i++ )
    {
        ra = a[i];

        rb =
                b[0]*fabs(R[i][0]) + b[1]*fabs(R[i][1]) + b[2]*fabs(R[i][2]);

        t = fabs(T[i]);

        if( t > ra + rb )
            return false;
    }

    //B's basis vectors
    for( k=0 ; k<3 ; k++ )
    {
        ra =
                a[0]*fabs(R[0][k]) + a[1]*fabs(R[1][k]) + a[2]*fabs(R[2][k]);
        rb = b[k];

        t =
                fabs( T[0]*R[0][k] + T[1]*R[1][k] +
                T[2]*R[2][k] );

        if( t > ra + rb )
            return false;
    }

    //9 cross products

    //L = A0 x B0
    ra =
            a[1]*fabs(R[2][0]) + a[2]*fabs(R[1][0]);

    rb =
            b[1]*fabs(R[0][2]) + b[2]*fabs(R[0][1]);

    t =
            fabs( T[2]*R[1][0] -
            T[1]*R[2][0] );

    if( t > ra + rb )
        return false;

    //L = A0 x B1
    ra =
            a[1]*fabs(R[2][1]) + a[2]*fabs(R[1][1]);

    rb =
            b[0]*fabs(R[0][2]) + b[2]*fabs(R[0][0]);

    t =
            fabs( T[2]*R[1][1] -
            T[1]*R[2][1] );

    if( t > ra + rb )
        return false;

    //L = A0 x B2
    ra =
            a[1]*fabs(R[2][2]) + a[2]*fabs(R[1][2]);

    rb =
            b[0]*fabs(R[0][1]) + b[1]*fabs(R[0][0]);

    t =
            fabs( T[2]*R[1][2] -
            T[1]*R[2][2] );

    if( t > ra + rb )
        return false;

    //L = A1 x B0
    ra =
            a[0]*fabs(R[2][0]) + a[2]*fabs(R[0][0]);

    rb =
            b[1]*fabs(R[1][2]) + b[2]*fabs(R[1][1]);

    t =
            fabs( T[0]*R[2][0] -
            T[2]*R[0][0] );

    if( t > ra + rb )
        return false;

    //L = A1 x B1
    ra =
            a[0]*fabs(R[2][1]) + a[2]*fabs(R[0][1]);

    rb =
            b[0]*fabs(R[1][2]) + b[2]*fabs(R[1][0]);

    t =
            fabs( T[0]*R[2][1] -
            T[2]*R[0][1] );

    if( t > ra + rb )
        return false;

    //L = A1 x B2
    ra =
            a[0]*fabs(R[2][2]) + a[2]*fabs(R[0][2]);

    rb =
            b[0]*fabs(R[1][1]) + b[1]*fabs(R[1][0]);

    t =
            fabs( T[0]*R[2][2] -
            T[2]*R[0][2] );

    if( t > ra + rb )
        return false;

    //L = A2 x B0
    ra =
            a[0]*fabs(R[1][0]) + a[1]*fabs(R[0][0]);

    rb =
            b[1]*fabs(R[2][2]) + b[2]*fabs(R[2][1]);

    t =
            fabs( T[1]*R[0][0] -
            T[0]*R[1][0] );

    if( t > ra + rb )
        return false;

    //L = A2 x B1
    ra =
            a[0]*fabs(R[1][1]) + a[1]*fabs(R[0][1]);

    rb =
            b[0] *fabs(R[2][2]) + b[2]*fabs(R[2][0]);

    t =
            fabs( T[1]*R[0][1] -
            T[0]*R[1][1] );

    if( t > ra + rb )
        return false;

    //L = A2 x B2
    ra =
            a[0]*fabs(R[1][2]) + a[1]*fabs(R[0][2]);

    rb =
            b[0]*fabs(R[2][1]) + b[1]*fabs(R[2][0]);

    t =
            fabs( T[1]*R[0][2] -
            T[0]*R[1][2] );

    if( t > ra + rb )
        return false;

    /*no separating axis found,
the two boxes overlap */

    return true;

}

void cpu_tsdf::RandomGenerateBoundingbox(const cpu_tsdf::TSDFHashing &tsdf_origin,
                                           const cpu_tsdf::OrientedBoundingBox& template_bb,
                                           const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
                                           cpu_tsdf::OrientedBoundingBox *bounding_boxes,
                                         const float similarity_thresh)
{
    using namespace std;
    cv::Vec3i min_pt, max_pt;
    // tsdf_origin.RecomputeBoundingBoxInVoxelCoord();
    tsdf_origin.getBoundingBoxInVoxelCoord(min_pt, max_pt);
    cv::Vec3i voxel_bb_size = max_pt - min_pt;
    cv::Vec3i template_bb_voxel_size = utility::EigenVectorToCvVector3(utility::ceil(
                template_bb.bb_sidelengths.cwiseQuotient(template_bb.voxel_lengths).eval()
                                                                       ));
    while (true)
    {
        // x1, y1: center
        int x1 = int( (float(rand())/RAND_MAX) * voxel_bb_size[0]) + min_pt[0];
        int y1 = int( (float(rand())/RAND_MAX) * voxel_bb_size[1]) + min_pt[1];
        int z1 = 0;
        //float theta = 0;
        float theta = int( (float(rand())/RAND_MAX) * (360.0/2.5) ) * 2.5 / 180.0 * M_PI;
 //       theta = 0;
//        cout << "maxpt: " << max_pt << endl;
//        cout << "minpt: " << min_pt << endl;
//        cout << "x1, y1: " << x1 << " " << y1 << endl;
//        cout << "template box: " << template_bb_voxel_size << endl;
//        cout << "template_voxl_length: " << template_bb.voxel_lengths << endl;
//        cout << "template side length:" << template_bb.bb_sidelengths << endl;
        // heuristic
        if (max_pt[0] - x1 < 0.5 * template_bb_voxel_size[0] ||
            max_pt[1] - y1 < 0.5 * template_bb_voxel_size[1] ||
            x1 - min_pt[0] < 0.5 * template_bb_voxel_size[0] ||
            y1 - min_pt[1] < 0.5 * template_bb_voxel_size[1])
            continue;

        Eigen::AngleAxisf rollAngle(theta, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotationMatrix = rollAngle.matrix();
        Eigen::Matrix3f cur_bb_orientation = rotationMatrix * Eigen::Matrix3f::Identity();
        Eigen::Vector3f cur_bb_sidelengths = template_bb.bb_sidelengths;
        Eigen::Vector3f cur_bb_voxel_lengths = template_bb.voxel_lengths;

        Eigen::Vector3f cur_bb_center = tsdf_origin.Voxel2World(Eigen::Vector3f(x1, y1, z1));
        Eigen::Vector3f cur_bb_offset = cur_bb_center - cur_bb_orientation * cur_bb_sidelengths/2.0;
        cur_bb_offset[2] = template_bb.bb_offset[2];

        OrientedBoundingBox cur_bb;
        cur_bb.bb_offset = cur_bb_offset;
        cur_bb.bb_orientation = cur_bb_orientation;
        cur_bb.bb_sidelengths = cur_bb_sidelengths;
        cur_bb.voxel_lengths = cur_bb_voxel_lengths;
        if(TestOBBsIntersection(cur_bb, avoided_bbs)) continue;
//        int intersect_sample;
//        float intersect_area;
//        if (TestOBBSimilarity(similarity_thresh, avoided_bbs, cur_bb, intersect_sample, intersect_area))
//        {
//            continue;
//        }

        *bounding_boxes = (cur_bb);
        break;
    }
}

//void cpu_tsdf::RandomGenerateBoundingboxes(const cpu_tsdf::TSDFHashing &tsdf_origin,
//                                           const cpu_tsdf::OrientedBoundingBox& template_bb, const int bb_num,
//                                           const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
//                                           std::vector<cpu_tsdf::OrientedBoundingBox> *bounding_boxes)
//{
//    Eigen::Vector3i voxel_bb_size;
//    tsdf_origin.getVoxelBoundingBoxSize(voxel_bb_size);
//    int gened_bb = 0;
//    while (gened_bb < bb_num)
//    {
//        int x1 = int( (float(rand())/RAND_MAX) * voxel_bb_size[0]);
//        int y1 = int( (float(rand())/RAND_MAX) * voxel_bb_size[1]);
//        float theta = int( (float(rand())/RAND_MAX) * (360.0/2.5) ) * 2.5 / 180.0 * M_PI;
//        int z1 = 0;
//        Eigen::Vector3f cur_bb_offset = tsdf_origin.Voxel2World(Eigen::Vector3f(x1, y1, z1));
//        cur_bb_offset[2] = template_bb[2];

//        Eigen::AngleAxisf rollAngle(theta, Eigen::Vector3f::UnitZ());
//        Eigen::Matrix3f rotationMatrix = rollAngle.matrix();
//        Eigen::Matrix3f cur_bb_orientation = rotationMatrix * Eigen::Matrix3f::Identity();

//        Eigen::Vector3f cur_bb_sidelengths = template_bb.bb_sidelengths;
//        Eigen::Vector3f cur_bb_voxel_lengths = template_bb.voxel_lengths;

//        OrientedBoundingBox cur_bb;
//        cur_bb.bb_offset = cur_bb_offset;
//        cur_bb.bb_orientation = cur_bb_orientation;
//        cur_bb.bb_sidelengths = cur_bb_sidelengths;
//        cur_bb.voxel_lengths = cur_bb_voxel_lengths;
//        if(TestOBBsIntersection(cur_bb, avoided_bbs)) continue;

//        bounding_boxes->push_back(cur_bb);
//        gened_bb++;
//    }
//}

//void cpu_tsdf::RandomGenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
//                                     const cpu_tsdf::OrientedBoundingBox& template_bb, const int bb_num,
//                                     const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
//                                     std::vector<cpu_tsdf::OrientedBoundingBox> *bounding_boxes,
//                                     std::vector<cpu_tsdf::TSDFFeature> *features,
//                                     std::vector<cpu_tsdf::TSDFHashing::Ptr> *feature_tsdfs)
//{
//    RandomGenerateBoundingboxes(tsdf_origin,
//                                               template_bb, bb_num,
//                                               avoided_bbs,
//                                               bounding_boxes);
//    GenerateSamples(tsdf_origin, *bounding_boxes, features, feature_tsdfs);
//}

void cpu_tsdf::RandomGenerateSamples(const cpu_tsdf::TSDFHashing &tsdf_origin,
                                     const cpu_tsdf::TSDFFeature& template_feature, const int bb_num,
                                     const std::vector<cpu_tsdf::OrientedBoundingBox> &avoided_bbs,
                                     std::vector<cpu_tsdf::TSDFFeature> *features,
                                     const std::string* save_path, float mesh_min_weight, float similarity_thresh)
{
    const cpu_tsdf::OrientedBoundingBox& template_bb = template_feature.GetOrientedBoundingBox();
    std::vector<float> out_feat_template;
    template_feature.GetFeatureVector(NULL, &out_feat_template, mesh_min_weight, true);
    const float template_occupied_ratio = template_feature.occupied_ratio(mesh_min_weight);
    int gened_bb = 0;
    while(gened_bb < bb_num)
    {
        printf("beginning current: %d th bb.\n", gened_bb);
        cpu_tsdf::OrientedBoundingBox cur_bb;
        cpu_tsdf::RandomGenerateBoundingbox(tsdf_origin,
                                                  template_bb,
                                                  avoided_bbs,
                                                  &cur_bb, similarity_thresh);
//        if (save_path)
//        {
//            char cur_bb_num[10];
//            sprintf(cur_bb_num, "%05d", gened_bb);
//            std::string save_filename = *save_path + "_debug_obb_" + cur_bb_num + ".ply";
//            cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, save_filename);
//        }

        cpu_tsdf::TSDFFeature cur_feat;
        if (cpu_tsdf::GenerateOneSample(tsdf_origin, cur_bb, template_occupied_ratio, mesh_min_weight, &cur_feat))
        {


//             cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//             SliceTSDFWithBoundingbox(&tsdf_origin, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
             // SliceTSDFWithBoundingbox(&tsdf_origin, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, tsdf_origin.voxel_length(), cur_feat_tsdf.get());

            // save them.
            //output the negative samples
            /********/
            cout << "save " << gened_bb << "th bb" << endl;
            char weight_num[10];
            sprintf(weight_num, "%.02f", mesh_min_weight);
            if (save_path)
            {
                using namespace std;
                const std::string& output_dir_prefix = *save_path;
                char number[10];
                sprintf(number, "_%05d", gened_bb);

//                cpu_tsdf::TSDFGridInfo grid_info(tsdf_origin,
//                                                tsdf_detection::VoxelSideLengthsFromVoxelLengths(cur_bb.bb_sidelengths, cur_bb.voxel_lengths),
//                                                 mesh_min_weight);
//                std::vector<cpu_tsdf::OrientedBoundingBox> obbs;
//                obbs.push_back(cur_bb);
//                cpu_tsdf::WriteObbsAndTSDFs(tsdf_origin, obbs, grid_info, output_dir_prefix + "_" + number);

//                string feat_name = output_dir_prefix + "_" + number + "_feat_" + weight_num + ".txt";
//                vector<float> neg_out_feat;
//                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                utility::OutputVector(feat_name, neg_out_feat);
//                assert(neg_out_feat.size() == out_feat_template.size());

////                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin.bin";
////                std::ofstream os(tsdf_bin_name, std::ios_base::out);
////                boost::archive::binary_oarchive oa(os);
////                oa << *(cur_feat_tsdf);

                string obb_name = output_dir_prefix + "_" + number + "_feat_obb.ply";
                cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);

//                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + ".ply";
//                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
            }
            /*******/
            if (features)
                features->push_back(std::move(cur_feat));
           // feature_tsdfs->push_back(cur_feat_tsdf);
            printf("end current: %d th bb.\n", gened_bb);
            gened_bb++;
        }
    }
}



//void cpu_tsdf::JitteringForOneOBB(const cpu_tsdf::OrientedBoundingBox &template_bb, int bb_num, std::vector<cpu_tsdf::OrientedBoundingBox> *jittered_bbs)
//{
//    const Eigen::Vector3f& bb_offset = template_bb.bb_offset;
//    const Eigen::Matrix3f& bb_orientation = template_bb.bb_orientation;

//    float theta = acos(bb_orientation(0,0));
//    const float dev_theta = M_PI/9; //M_PI/3/3, 3-sigma?

//    float origin_x  = bb_offset.x;
//    const float dev_x = 2/3;

//    float origin_y = bb_offset.y;
//    const float dev_y = 2/3;

//    std::normal_distribution<float> theta_distribution(theta, dev_theta);
//    std::normal_distribution<float> x_distribution(x, dev_x);
//    std::normal_distribution<float> y_distribution(y, dev_y);


//}

void cpu_tsdf::GenerateSamplesJittering(const cpu_tsdf::TSDFHashing &tsdf_origin,
                                        const cpu_tsdf::TSDFFeature& template_feature,
                                        int bb_num,
                                        std::vector<cpu_tsdf::TSDFFeature> *features,
                                        const std::string *save_path, float mesh_min_weight)
{
    //std::vector<float> out_feat_template;
    //template_feature.GetFeatureVector(NULL, &out_feat_template, mesh_min_weight, true);

    const cpu_tsdf::OrientedBoundingBox &template_bb = template_feature.GetOrientedBoundingBox();
    const float template_occupy_ratio = template_feature.occupied_ratio(mesh_min_weight);
    const Eigen::Vector3f& bb_offset = template_bb.bb_offset;
    const Eigen::Matrix3f& bb_orientation = template_bb.bb_orientation;

    Eigen::AngleAxisf orig_angleaxis(bb_orientation);
    // float theta = orig_angleaxis.angle() * orig_angleaxis.axis()[2];
    float theta = orig_angleaxis.angle();
    CHECK_LT(fabs(fabs(orig_angleaxis.axis()[2]) - 1), 1e-3);
    //tsdf_detection::NormalizeAngle(theta);
    // const float dev_theta = M_PI/18; //M_PI/3/3, 3-sigma?
    const float dev_theta = M_PI/48; //try uniform distribution

    float origin_x  = bb_offset[0];
    const float dev_x = template_bb.bb_sidelengths[0]/30;

    float origin_y = bb_offset[1];
    const float dev_y = template_bb.bb_sidelengths[1]/30;

     std::default_random_engine generator;
    // std::normal_distribution<float> theta_distribution(theta, dev_theta);
    std::uniform_real_distribution<float> theta_distribution(theta - dev_theta, theta + dev_theta);
    std::normal_distribution<float> x_distribution(0, dev_x);
    std::normal_distribution<float> y_distribution(0, dev_y);

    //cpu_tsdf::WriteOrientedBoundingboxPly(template_bb, *save_path + "template.ply");
    int gened_bb = 0;
    while(gened_bb < bb_num)
    {
        printf("beginning current: %d th bb.\n", gened_bb);
        cpu_tsdf::OrientedBoundingBox cur_bb;
        Eigen::Vector3f obb_center = template_bb.BoxCenter();
//        Eigen::Vector3f obb_center = template_bb.bb_offset + template_bb.bb_orientation.col(0) * template_bb.bb_sidelengths[0]/2.0f +
//                template_bb.bb_orientation.col(1) * template_bb.bb_sidelengths[1]/2.0f +
//                template_bb.bb_orientation.col(2) * template_bb.bb_sidelengths[2]/2.0f;
        Eigen::Vector3f obb_jittered = obb_center + Eigen::Vector3f(x_distribution(generator),
                                                                    y_distribution(generator),
                                                                    0);
        Eigen::AngleAxisf rollAngle(theta_distribution(generator), orig_angleaxis.axis());
        //Eigen::AngleAxisf rollAngle(theta, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotationMatrix = rollAngle.matrix();
        cur_bb.bb_orientation = rotationMatrix;
        cur_bb.bb_sidelengths = template_bb.bb_sidelengths;
        cur_bb.voxel_lengths = template_bb.voxel_lengths;
        cur_bb.bb_offset = obb_jittered - cur_bb.bb_orientation.col(0) * cur_bb.bb_sidelengths[0]/2.0f -
                cur_bb.bb_orientation.col(1) * cur_bb.bb_sidelengths[1]/2.0f -
                cur_bb.bb_orientation.col(2) * cur_bb.bb_sidelengths[2]/2.0f;

        if (save_path)
        {
            char cur_bb_num[10];
            sprintf(cur_bb_num, "%05d", gened_bb);
            std::string save_filename = *save_path + "_jiterring_debug_obb_" + cur_bb_num + ".ply";
            cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, save_filename);
        }

        cpu_tsdf::TSDFFeature cur_feat;
        if (cpu_tsdf::GenerateOneSample(tsdf_origin, cur_bb, template_occupy_ratio, mesh_min_weight, &cur_feat))
        {
             //cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
             //SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_origin, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
            // save them.
            //output the negative samples
            /********/
            cout << "save " << gened_bb << "th bb" << endl;
            char weight_num[10];
            sprintf(weight_num, "%.02f", mesh_min_weight);
            if (save_path)
            {
                using namespace std;
                const std::string& output_dir_prefix = *save_path;
                char number[10];
                sprintf(number, "_%05d", gened_bb);
//                string feat_name = output_dir_prefix + "_" + number + "_feat_jittering" + weight_num + ".txt";
//                vector<float> neg_out_feat;
//                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                utility::OutputVector(feat_name, neg_out_feat);
                //assert(neg_out_feat.size() == out_feat_template.size());

//                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_jittering.bin";
//                std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                boost::archive::binary_oarchive oa(os);
//                oa << *(cur_feat_tsdf);

                string obb_name = output_dir_prefix + "_" + number + "_feat_obb_jittering.ply";
                cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);

//                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_jittering_" + weight_num + ".ply";
//                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
            }
            /*******/
//            if (bounding_boxes)
//                bounding_boxes->push_back(cur_bb);
            if (features)
                features->push_back(std::move(cur_feat));
           // feature_tsdfs->push_back(cur_feat_tsdf);
            printf("end current: %d th bb.\n", gened_bb);
            gened_bb++;
        }
    }

}

//void cpu_tsdf::GenerateSamplesJitteringNew(const cpu_tsdf::TSDFHashing &tsdf_origin,
//                                        const cpu_tsdf::TSDFFeature& template_feature,
//                                        int bb_num,
//                                        std::vector<cpu_tsdf::TSDFFeature> *features,
//                                        const std::string *save_path, float mesh_min_weight)
//{
//    cout << "begin jiter samples" << endl;
//    const cpu_tsdf::OrientedBoundingBox &template_bb = template_feature.GetOrientedBoundingBox();
//    const float template_occupy_ratio = template_feature.occupied_ratio(mesh_min_weight);
//    const Eigen::Vector3f bb_bottom_center = template_bb.BoxBottomCenter();
//    const Eigen::Matrix3f bb_orientation =template_bb.bb_orientation;
//    const Eigen::Vector3f bb_sidelengths = template_bb.bb_sidelengths;
//    const Eigen::Vector3i bb_voxel_sidelengths = template_feature.VoxelSideLengths();

//    std::default_random_engine generator;
//    float theta = acos(bb_orientation(0,0));
//    // const float dev_theta = M_PI/18; //M_PI/3/3, 3-sigma?
//    const float dev_theta = M_PI/24; //try uniform distribution
//    std::uniform_real_distribution<float> theta_distribution(theta - dev_theta, theta + dev_theta);
//    // std::normal_distribution<float> theta_distribution(theta, dev_theta);
//    float origin_centerx  = bb_bottom_center[0];
//    const float dev_x = 0.8;
//    std::normal_distribution<float> x_distribution(origin_centerx, dev_x);
//    float origin_centery = bb_bottom_center[1];
//    const float dev_y = 0.8;
//    std::normal_distribution<float> y_distribution(origin_centery, dev_y);
//    float dev_scalex = 0.01;
//    float dev_scaley = 0.01;
//    float dev_scalez = 0.01;
//    std::normal_distribution<float> scale_x_distribution(1, dev_scalex);
//    std::normal_distribution<float> scale_y_distribution(1, dev_scaley);
//    std::normal_distribution<float> scale_z_distribution(1, dev_scalez);

//    int gened_bb = 0;
//    cout << bb_num << endl;
//    while(gened_bb < bb_num)
//    {
//        printf("beginning current: %d th bb.\n", gened_bb);
//        float cur_centerx = x_distribution(generator);
//        float cur_centery = y_distribution(generator);
//        float cur_angle = theta_distribution(generator);
//        float cur_scalex = scale_x_distribution(generator);
//        if (cur_scalex < 0.95) continue;
//        float cur_scaley = scale_y_distribution(generator);
//        if (cur_scaley < 0.95) continue;
//        float cur_scalez = scale_z_distribution(generator);
//        if (cur_scalez < 0.9) continue;
////        float cur_scalex = 1;
////        if (cur_scalex < 0.95) continue;
////        float cur_scaley = 1;
////        if (cur_scaley < 0.95) continue;
////        float cur_scalez = 1;
////        if (cur_scalez < 0.9) continue;
//        OrientedBoundingBox cur_bb = cpu_tsdf::ComputeOBBFromBottomCenter(
//                    Eigen::Vector2f(cur_centerx, cur_centery),
//                    bb_bottom_center[2],
//                    Eigen::AngleAxisf(cur_angle, Eigen::Vector3f::UnitZ()).matrix(),
//                    bb_sidelengths.cwiseProduct(Eigen::Vector3f(cur_scalex, cur_scaley, cur_scalez)),
//                    bb_voxel_sidelengths
//                    );

////        if (save_path)
////        {
////            char cur_bb_num[10];
////            sprintf(cur_bb_num, "%05d", gened_bb);
////            std::string save_filename = *save_path + "_jiterring_debug_obb_" + cur_bb_num + ".ply";
////            cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, save_filename);
////        }

//        //cout << "begin generate feature" << endl;
//        cpu_tsdf::TSDFFeature cur_feat;
//        if (cpu_tsdf::GenerateOneSample(tsdf_origin, cur_bb, template_occupy_ratio, mesh_min_weight, &cur_feat))
//        {
//             //cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//             //SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_origin, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//            // save them.
//            //output the negative samples
//            /********/
//            cout << "save " << gened_bb << "th bb" << endl;
//            char weight_num[10];
//            sprintf(weight_num, "%.02f", mesh_min_weight);
//            if (save_path)
//            {
//                using namespace std;
//                const std::string& output_dir_prefix = *save_path;
//                char number[10];
//                sprintf(number, "_%05d", gened_bb);
////                string feat_name = output_dir_prefix + "_" + number + "_feat_jittering" + weight_num + ".txt";
////                vector<float> neg_out_feat;
////                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
////                utility::OutputVector(feat_name, neg_out_feat);
//                //assert(neg_out_feat.size() == out_feat_template.size());

////                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_jittering.bin";
////                std::ofstream os(tsdf_bin_name, std::ios_base::out);
////                boost::archive::binary_oarchive oa(os);
////                oa << *(cur_feat_tsdf);

//                string obb_name = output_dir_prefix + "_" + number + "_feat_obb_jittering.ply";
//                cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);

////                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_jittering_" + weight_num + ".ply";
////                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
//            }
//            /*******/
////            if (bounding_boxes)
////                bounding_boxes->push_back(cur_bb);
//            if (features)
//                features->push_back(std::move(cur_feat));
//           // feature_tsdfs->push_back(cur_feat_tsdf);
//            printf("end current: %d th bb.\n", gened_bb);
//            gened_bb++;
//        }
//    }

//}


void cpu_tsdf::TSDFFeature::ComputeFeature(const OrientedBoundingBox &vobb, const cpu_tsdf::TSDFHashing &tsdf_origin)
{
    this->Clear();
    this->obb = vobb;
    Eigen::Vector3i voxel_size = VoxelSideLengths();
    int total_voxel_size = voxel_size[0] * voxel_size[1] * voxel_size[2];
    this->occupied_.resize(total_voxel_size);
    Eigen::Matrix3f bb_orientation_scaled = obb.bb_orientation * obb.voxel_lengths.asDiagonal();
    int cur_cnt = 0;
    for (int ix = 0; ix < voxel_size[0]; ++ix)
        for (int iy = 0; iy < voxel_size[1]; ++iy)
            for (int iz = 0; iz < voxel_size[2]; ++iz)
            {
                Eigen::Vector3f cur_world_coord = obb.bb_offset + (bb_orientation_scaled.col(0) * ix + bb_orientation_scaled.col(1) * iy + bb_orientation_scaled.col(2) * iz);
                //std::cerr << "world_pt: " << cur_world_coord << std::endl;
                float cur_d = 0;
                float cur_w = 0;
     //   if (ix == 5 && iy == 7 && iz == 4) {
     //       std::cerr << "old trans world_pt 11-6-2: " << cur_world_coord << std::endl;
     //   }
                if (tsdf_origin.RetriveDataFromWorldCoord(cur_world_coord, &cur_d, &cur_w))
                // if (tsdf_origin.RetriveDataFromWorldCoord_NearestNeighbor(cur_world_coord, &cur_d, &cur_w))
                {
                    this->occupied_[cur_cnt] = true;
                    this->tsdf_vals.push_back(cur_d);
                    this->tsdf_weights.push_back(cur_w);
                    //std::cerr << "world_pt: " << cur_world_coord << " cur_d: " << cur_d << " cur_w: " << cur_w << std::endl;
     //   if (ix == 5 && iy == 7 && iz == 4) {
     //   std::cerr << "old trans world_pt 11-6-2: " << cur_world_coord << std::endl;
     //   std::cerr << "old world_pt: 11-6-2 " << cur_world_coord << " cur_d: " << cur_d << " cur_w: " << cur_w << std::endl;
     //   }
                }
                else
                {
                    this->occupied_[cur_cnt] = false;
     //   if (ix == 5 && iy == 7 && iz == 4) {
     //   std::cerr << "old trans world_pt 11-6-2: " << cur_world_coord << std::endl;
     //   std::cerr << "old world_pt: 11-6-2 " << cur_world_coord << " cur_d: " << cur_d << " cur_w: " << cur_w << std::endl;
     //   }
                }
                cur_cnt ++;
            }
}

//void cpu_tsdf::SlidingBoxDetection(const cpu_tsdf::TSDFHashing &tsdf_model,
//                                   const cpu_tsdf::OrientedBoundingBox &template_bb, SVMWrapper &svm,
//                                   const float delta_x, const float delta_y, const float delta_rotation,
//                                   float *x_st_ed, float *y_st_ed, std::vector<ClassificationResult> *res,
//                                   float mesh_min_weight,  const std::string* save_path)
//{
//    FILE* hf = NULL;
//    if (save_path)
//    {
//        hf = fopen(((*save_path) + "_SlidingBoxDetectionResults.txt").c_str(), "w");
//        fprintf(hf, "index label score\n");
//    }
//    using namespace std;
//    Eigen::Vector3f min_pt, max_pt;
//    tsdf_model.getBoundingBoxInWorldCoord(min_pt, max_pt);
//    Eigen::Vector3f box_center = template_bb.BoxCenter();
//    cpu_tsdf::TSDFFeature template_feature;
//    template_feature.ComputeFeature(template_bb.bb_orientation, template_bb.bb_sidelengths, template_bb.bb_offset, template_bb.voxel_lengths,
//                                    tsdf_model);

//    x_st_ed[0] = min_pt[0];
//    x_st_ed[1] = max_pt[0];

//    y_st_ed[0] = min_pt[1];
//    y_st_ed[1] = max_pt[1];

////    res->clear();
//    cout << "x sample num: " << floor((max_pt[0] - min_pt[0])/delta_x) + 1  << endl;
//    cout << "y sample num: " << floor((max_pt[1] - min_pt[1])/delta_y) + 1  << endl;
//    cout << "rotate sample num: " << floor((2*M_PI - 1e-5)/delta_rotation) + 1 << endl;
//    int x_s_num = floor((max_pt[0] - min_pt[0])/delta_x) + 1;
//    int y_s_num = floor((max_pt[1] - min_pt[1])/delta_y) + 1;
//    int rotate_s_num = floor((2*M_PI - 1e-5)/delta_rotation) + 1;
//    fprintf(hf, "min_pt: %f %f %f\n", min_pt[0], min_pt[1], min_pt[2]);
//    fprintf(hf, "max_pt: %f %f %f\n", max_pt[0], max_pt[1], max_pt[2]);
//    fprintf(hf, "delta_x, delta_y, delta_r: %f %f %f\n", delta_x, delta_y, delta_rotation);
//    int box_index = -1;

//    res->resize(x_s_num * y_s_num * rotate_s_num);
//    for (float curx = min_pt[0]; curx < max_pt[0]; curx += delta_x)
//        for (float cury = min_pt[1]; cury < max_pt[1]; cury += delta_y)
//            for (float rotate = 0; rotate < 2 * M_PI - 1e-5; rotate += delta_rotation)
//            {
//                box_index++;
//                cout << "box: " << box_index << endl;
/////debug
////                float debugx, debugy, debugang;
////                BoxIndexToBoxPos(x_st_ed,  y_st_ed,
////                                      delta_x, delta_y, delta_rotation,
////                                      box_index, &debugx,  &debugy, &debugang);
////                assert(fabs(debugx - curx) < 1e-3 && fabs(debugy - cury) < 1e-3 && fabs(debugang - rotate) < 1e-3);
////                int box_ind;
////                BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, curx, cury, rotate, &box_ind);
////                assert(box_ind == box_index);

//                cpu_tsdf::OrientedBoundingBox cur_bb;
//                Eigen::Vector3f cur_center = Eigen::Vector3f(curx, cury, box_center[2]);
//                GenerateBoundingbox(template_bb, cur_center, rotate, &cur_bb);
////                if (save_path)
////                {
////                    const std::string& output_dir_prefix = *save_path;
////                    char number[10];
////                    sprintf(number, "_%05d", box_index);
////                    string obb_name = output_dir_prefix + "_" + number + "_feat_debug_all_obb.ply";
////                    cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);
////                }

//                cpu_tsdf::TSDFFeature cur_feat;
//                if (cpu_tsdf::GenerateOneSample(tsdf_model, cur_bb, template_feature, &cur_feat, mesh_min_weight))
//                {
//                     cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                     SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                     ////////
//                    vector<vector<float>> feat_vec(1);
//                    cur_feat.GetFeatureVector(&template_feature,  &(feat_vec[0]), mesh_min_weight, true);
//                    vector<int> input_labels(1, -1);
//                    vector<int> output_label;
//                    vector<vector<float>> output_score;
//                    float accuracy;
//                    svm.SVMPredict(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
//                    int cur_label = output_label[0];
//                    float cur_score = output_score[0][0];
//                    fprintf(hf, "%d %d %f\n", box_index, cur_label, cur_score);
//                    fflush(hf);

//                    ////////////////////////////
//                    if ( cur_score >= -0.5)
//                    {
//                        //output the hard negative samples
//                        if (save_path)
//                        {
//                            /********/
//                            cout << "save " << box_index << "th bb" << endl;
//                            char weight_num[10];
//                            sprintf(weight_num, "%.02f", mesh_min_weight);
//                            if (save_path)
//                            {
//                                using namespace std;
//                                char cscore[15];
//                                char clabel[5];
//                                sprintf(cscore, "%.04f", cur_score);
//                                sprintf(clabel, "%02d", cur_label);
//                                const std::string output_dir_prefix = *save_path + "_detected_label_" + clabel + "_score_" + cscore;
//                                char number[10];
//                                sprintf(number, "_%05d", box_index);
//                                string feat_name = output_dir_prefix + "_boxidx_" + number + "_feat_" + weight_num + ".txt";
//                                vector<float> neg_out_feat;
//                                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                                utility::OutputVector(feat_name, neg_out_feat);
//                                //assert(neg_out_feat.size() == out_feat_template.size());

//                                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin.bin";
//                                std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                boost::archive::binary_oarchive oa(os);
//                                oa << *(cur_feat_tsdf);

//                                string obb_name = output_dir_prefix + "_" + number + "_feat_obb.ply";
//                                cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);

//                                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + ".ply";
//                                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
//                            }
//                            /*******/
//                        }
//                    }
//                    ////////////////////////////
//                    (*res)[box_index] = (ClassificationResult(box_index, cur_label, cur_score));
//                }
//            }
//    if (save_path)
//    {
//        fclose(hf);
//    }
//    return;
//}

//void cpu_tsdf::SlidingBoxDetectionReturningHardNegative(const cpu_tsdf::TSDFHashing &tsdf_model,
//                                                        const cpu_tsdf::OrientedBoundingBox &template_bb,
//                                                        SVMWrapper &svm,
//                                                        const float delta_x, const float delta_y, const float delta_rotation,
//                                                        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
//                                                        const std::vector<std::vector<float>>& pos_feats,
//                                                        const std::vector<cpu_tsdf::OrientedBoundingBox>& cached_bbs,
//                                                        std::vector<cpu_tsdf::OrientedBoundingBox>* hard_neg_bbs,
//                                                        std::vector<std::vector<float>>* hard_negatives,
//                                                        float similar_tresh,
//                                                        float mesh_min_weight,  const std::string* save_path)
//{
//    FILE* hf = NULL;
//    if (save_path)
//    {
//        hf = fopen(((*save_path) + "_SlidingBoxHardNeg.txt").c_str(), "w");
//        fprintf(hf, "index label score\n");
//    }
//    using namespace std;
//    Eigen::Vector3f min_pt, max_pt;
//    tsdf_model.getBoundingBoxInWorldCoord(min_pt, max_pt);
//    Eigen::Vector3f box_center = template_bb.BoxCenter();
//    cpu_tsdf::TSDFFeature template_feature;
//    template_feature.ComputeFeature(template_bb.bb_orientation, template_bb.bb_sidelengths, template_bb.bb_offset, template_bb.voxel_lengths,
//                                    tsdf_model);

////    x_st_ed[0] = min_pt[0];
////    x_st_ed[1] = floor((max_pt[0] - min_pt[0])/delta_x) * delta_x + min_pt[0];

////    y_st_ed[0] = min_pt[0];
////    y_st_ed[1] = floor((max_pt[1] - min_pt[1])/delta_y) * delta_y + max_pt[1];

////    res->clear();
//    cout << "x sample num: " << floor((max_pt[0] - min_pt[0])/delta_x) + 1  << endl;
//    cout << "y sample num: " << floor((max_pt[1] - min_pt[1])/delta_y) + 1  << endl;
//    cout << "rotate sample num: " << floor((2*M_PI - 1e-5)/delta_rotation) + 1 << endl;
//    fprintf(hf, "min_pt: %f %f %f\n", min_pt[0], min_pt[1], min_pt[2]);
//    fprintf(hf, "max_pt: %f %f %f\n", max_pt[0], max_pt[1], max_pt[2]);
//    fprintf(hf, "delta_x, delta_y, delta_r: %f %f %f\n", delta_x, delta_y, delta_rotation);
//    int box_index = -1;

//    int gt_pos_num = pos_feats.size();
//    std::vector<std::vector<float>> normed_pos_tsdf_feats(gt_pos_num);
//    for (int i = 0; i < gt_pos_num; ++i)
//    {
//        normed_pos_tsdf_feats[i] = NormalizeVector(pos_feats[i]);
//    }
//    for (float curx = min_pt[0]; curx < max_pt[0]; curx += delta_x)
//        for (float cury = min_pt[1]; cury < max_pt[1]; cury += delta_y)
//            for (float rotate = 0; rotate < 2 * M_PI - 1e-5; rotate += delta_rotation)
//            {
//                box_index++;
//                cout << "box: " << box_index << endl;

//                cpu_tsdf::OrientedBoundingBox cur_bb;
//                Eigen::Vector3f cur_center = Eigen::Vector3f(curx, cury, box_center[2]);
//                GenerateBoundingbox(template_bb, cur_center, rotate, &cur_bb);
////                if (save_path)
////                {
////                    const std::string& output_dir_prefix = *save_path;
////                    char number[10];
////                    sprintf(number, "_%05d", box_index);
////                    string obb_name = output_dir_prefix + "_" + number + "_feat_debug_all_obb.ply";
////                    cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);
////                }
//                // only sample negative samples
//                //if(TestOBBsIntersection(cur_bb, pos_bbs)) continue;
//                if (OBBsLargeOverlapArea(cur_bb, pos_bbs)) continue;
//                if(TestOBBsCompletelyOverlap(cur_bb, cached_bbs)) continue;

//                cpu_tsdf::TSDFFeature cur_feat;
//                if (cpu_tsdf::GenerateOneSample(tsdf_model, cur_bb, template_feature, &cur_feat, mesh_min_weight))
//                {
//                     cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                     SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                     ////////
//                    vector<vector<float>> feat_vec(1);
//                    cur_feat.GetFeatureVector(&template_feature,  &(feat_vec[0]), mesh_min_weight, true);
//                    vector<int> input_labels(1, -1);
//                    vector<int> output_label;
//                    vector<vector<float>> output_score;
//                    float accuracy;
//                    svm.SVMPredict(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
//                    int cur_label = output_label[0];
//                    float cur_score = output_score[0][0];
//                    fprintf(hf, "%d %d %f\n", box_index, cur_label, cur_score);

//                    ////////////////////////////
//                    int gt_label = -1; // only sampling negative samples, so the ground truth should be -1.
//                    if ( gt_label * output_score[0][0] < 1 /* within margin or wrong*/
//                         && !TestIsVectorSimilar(normed_pos_tsdf_feats, feat_vec[0], similar_tresh) /* not similar with any pos sample */
//                         )
//                    {
//                        /*it's hard negative*/
//                        //output the hard negative samples
//                        if (save_path)
//                        {
//                            /********/
//                            cout << "save " << box_index << "th bb" << endl;
//                            char weight_num[10];
//                            sprintf(weight_num, "%.02f", mesh_min_weight);
//                            if (save_path)
//                            {
//                                using namespace std;
//                                char cscore[15];
//                                char clabel[5];
//                                sprintf(cscore, "%.04f", cur_score);
//                                sprintf(clabel, "%02d", cur_label);
//                                const std::string output_dir_prefix = *save_path + "_hard_neg_label_" + clabel + "_score_" + cscore;
//                                char number[10];
//                                sprintf(number, "_%05d", box_index);
//                                string feat_name = output_dir_prefix + "_" + number + "_feat_" + weight_num + ".txt";
//                                vector<float> neg_out_feat;
//                                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                                utility::OutputVector(feat_name, neg_out_feat);
//                                //assert(neg_out_feat.size() == out_feat_template.size());

//                                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin.bin";
//                                std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                boost::archive::binary_oarchive oa(os);
//                                oa << *(cur_feat_tsdf);

//                                string obb_name = output_dir_prefix + "_" + number + "_feat_obb.ply";
//                                cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);

//                                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + ".ply";
//                                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
//                            }
//                            /*******/
//                        }
//                        if (hard_neg_bbs) hard_neg_bbs->push_back(cur_bb);
//                        if (hard_negatives) hard_negatives->push_back(feat_vec[0]);
//                    }
//                    ////////////////////////////
//                    //res->push_back(ClassificationResult(box_index, cur_label, cur_score));
//                }
//            }
//    if (save_path)
//    {
//        fclose(hf);
//    }
//    return;
//}


void cpu_tsdf::GenerateBoundingbox(const cpu_tsdf::OrientedBoundingBox &template_bb, const Eigen::Vector3f &center_pos_world, const float theta, cpu_tsdf::OrientedBoundingBox *bounding_box)
{
    //Eigen::AngleAxisf rollAngle(theta, Eigen::Vector3f::UnitZ());
    //Eigen::Matrix3f rotationMatrix = rollAngle.matrix();
    Eigen::Matrix3f rotationMatrix;
    rotationMatrix << cos(theta), -sin(theta), 0,
            sin(theta), cos(theta), 0,
            0,0,1;
    Eigen::Matrix3f cur_bb_orientation = rotationMatrix * template_bb.bb_orientation;
    Eigen::Vector3f cur_bb_sidelengths = template_bb.bb_sidelengths;
    Eigen::Vector3f cur_bb_voxel_lengths = template_bb.voxel_lengths;

    Eigen::Vector3f cur_bb_center = center_pos_world;
    Eigen::Vector3f cur_bb_offset = cur_bb_center - cur_bb_orientation * cur_bb_sidelengths/2.0;
    //cur_bb_offset[2] = template_bb.bb_offset[2];

    OrientedBoundingBox cur_bb;
    cur_bb.bb_offset = cur_bb_offset;
    cur_bb.bb_orientation = cur_bb_orientation;
    cur_bb.bb_sidelengths = cur_bb_sidelengths;
    cur_bb.voxel_lengths = cur_bb_voxel_lengths;

    *bounding_box = (cur_bb);
}

bool RemoveRepeatedNegSamples(
        const int positive_num,
        std::vector<std::vector<float>>* feature_cache,
        std::vector<int>* label_cache,
        std::vector<cpu_tsdf::OrientedBoundingBox>* bbs_cache)
{
    using namespace std;
    cerr << "begin remove repeated hard negs" << endl;
    Timer t;
    t.start("start remove rep");
    struct SortStruct
    {
        Eigen::Vector3f xyangle;
        int original_idx;
    };
    const int feat_size = feature_cache->size();
    std::vector<SortStruct> sort_vec(bbs_cache->size() - positive_num);
    for (int i = positive_num; i < bbs_cache->size(); ++i)
    {
        SortStruct cur_item;
        tsdf_detection::OBBToXYAngle((*bbs_cache)[i], &(cur_item.xyangle));
        cur_item.original_idx = i;
        sort_vec[i - positive_num] = cur_item;
    }
    tsdf_detection::XYAngleLess lesscmp;
    std::sort(sort_vec.begin(), sort_vec.end(),
              [lesscmp](const SortStruct& lhs, const SortStruct& rhs)
    {
        return lesscmp(lhs.xyangle, rhs.xyangle);
    } );

     tsdf_detection::XYAngleEqual equalcmp;
    auto last = std::unique(sort_vec.begin(), sort_vec.end(),
                [equalcmp](const SortStruct& lhs, const SortStruct& rhs)
      {

        return equalcmp(lhs.xyangle, rhs.xyangle);
       });
    t.start("after unique");

    std::vector<vector<float>> new_feature_cache(feature_cache->begin(), feature_cache->begin() + positive_num);
    std::vector<int> new_label_cache(label_cache->begin(), label_cache->begin() + positive_num);
    std::vector<cpu_tsdf::OrientedBoundingBox> new_bbs_cache(bbs_cache->begin(), bbs_cache->begin() + positive_num);
    for (auto itr = sort_vec.begin(); itr != last; ++itr)
    {
        new_feature_cache.push_back((*feature_cache)[itr->original_idx]);
        new_label_cache.push_back((*label_cache)[itr->original_idx]);
        new_bbs_cache.push_back((*bbs_cache)[itr->original_idx]);
    }
    const int cur_feat_size = new_feature_cache.size();
    t.stop();
    t.plot();
    cerr << "Remove rep neg sample: display prev feat_size/cur_feat_size: \n" << feat_size << " / " << cur_feat_size << endl;
    CHECK_EQ(new_feature_cache.size(), new_label_cache.size());
    CHECK_EQ(new_bbs_cache.size(), new_feature_cache.size());
    *feature_cache = new_feature_cache;
    *label_cache = new_label_cache;
    *bbs_cache = new_bbs_cache;
    return true;
}

int cpu_tsdf::TrainObjectDetector(const cpu_tsdf::TSDFHashing &tsdf_model,
                                  const cpu_tsdf::OrientedBoundingBox &template_obb,
                                  const std::vector<cpu_tsdf::OrientedBoundingBox>& positive_sample_obbs,
                                  const std::vector<bool>& positive_sample_for_training,
                                  int bb_jitter_num_each,
                                  int random_sample_num,
                                  SVMWrapper *svm,
                                  const std::string* save_path,
                                  float similarity_thresh,
                                  float mesh_min_weight,
                                  const Eigen::Vector3f& detection_deltas,
                                  const int total_thread,
                                  const Eigen::Vector3f& scene_min_pt,
                                  const Eigen::Vector3f& scene_max_pt,
                                  const std::string& train_options,
                                  const std::string& predict_options,
                                  // temporary
                                  std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
                                  std::vector<cpu_tsdf::OrientedBoundingBox>& neg_bbs
                                  )
{
    const int num_positive_sample_training  = std::count(positive_sample_for_training.begin(), positive_sample_for_training.end(), true);
    float x_st_ed[2] = {scene_min_pt[0], scene_max_pt[0]};
    float y_st_ed[2] = {scene_min_pt[1], scene_max_pt[1]};

//    using namespace std;
//    const float svm_param_c = 1; //0.01
//    const float svm_param_w1 = 10; //50
//    char train_options[256];
//    sprintf(train_options, "-s 0 -t 0 -c %f -w1 %.9f ", svm_param_c, svm_param_w1);
//    char predict_options[256] = {'\0'};


    cpu_tsdf::WriteOrientedBoundingboxPly(template_obb, *save_path + "original_template_for_train.ply");
    using namespace std;
    cpu_tsdf::TSDFFeature template_feature;
    template_feature.ComputeFeature(template_obb, tsdf_model);

    vector<vector<float>> features_cache;
    vector<int> labels_cache;
    vector<cpu_tsdf::OrientedBoundingBox> bbs_cache;
    int pos_sample_num = 0;
    // std::vector<std::vector<float>> pos_features_vecs;
    std::vector<cpu_tsdf::OrientedBoundingBox> pos_all_bbs;
    {
        // 0. init sample cache
        // sample positive samples
        //const int max_pos_to_use = positive_sample_obbs.size();
        std::vector<cpu_tsdf::TSDFFeature> pos_all_feats(num_positive_sample_training);
        std::vector<cpu_tsdf::OrientedBoundingBox> pos_bb_to_use(num_positive_sample_training);
        int cnt = 0;
        for (int i = 0; i < positive_sample_obbs.size(); ++i)
        {
            if (positive_sample_for_training[i])
            {
                pos_all_feats[cnt].ComputeFeature(positive_sample_obbs[i], tsdf_model);
                pos_bb_to_use[cnt] = positive_sample_obbs[i];
                cnt++;
            }
        }
        cpu_tsdf::WriteOrientedBoundingboxesPly(pos_bb_to_use, *save_path + "original_pos_sample_for_train.ply");
//        pos_bb_to_use.resize(1);
//        pos_bb_to_use[0] = positive_sample_obbs[0];
        // GenerateSamplesJitteringForMultipleBB(tsdf_model, positive_sample_obbs, bb_jitter_num_each, &pos_all_bbs, &pos_all_feats, save_path, mesh_min_weight);
        GenerateSamplesJitteringForMultipleBB(tsdf_model, pos_bb_to_use, bb_jitter_num_each, &pos_all_feats, save_path, mesh_min_weight);
        pos_sample_num = pos_all_feats.size();

        std::vector<cpu_tsdf::TSDFFeature> neg_feats;
        // sample negtive samples
        string cur_save_path;
        if (save_path)
        {
            cur_save_path = *save_path + "_neg";
        }
        //RandomGenerateSamples(tsdf_model, template_obb, random_sample_num, template_feature, positive_sample_obbs, &neg_bbs,  &neg_feats, save_path ? &cur_save_path:NULL, mesh_min_weight);
        RandomGenerateSamples(tsdf_model, template_feature, random_sample_num, positive_sample_obbs, &neg_feats, save_path ? &cur_save_path:NULL, mesh_min_weight, similarity_thresh);

        // generate obb cache
        // bbs_cache.assign(pos_all_bbs.begin(), pos_all_bbs.end());
        // bbs_cache.insert(bbs_cache.end(), neg_bbs.begin(), neg_bbs.end());
        // generate feature cache
        AddTSDFFeatureToFeatVec(pos_all_feats, template_feature, &features_cache, &bbs_cache, mesh_min_weight);
        // save positive features, later we compare hard negative features to this to decide real hard negatives
        // pos_features_vecs = features_cache;
        pos_all_bbs = bbs_cache;
        // zctemp
        pos_bbs = pos_all_bbs;

        AddTSDFFeatureToFeatVec(neg_feats, template_feature, &features_cache, &bbs_cache, mesh_min_weight);
        // zctemp
        neg_bbs.assign(bbs_cache.begin() + pos_all_bbs.size(), bbs_cache.end());

        labels_cache.assign(pos_all_feats.size() + neg_feats.size(), -1);
        std::fill(labels_cache.begin(), labels_cache.begin() + pos_all_feats.size(), 1);
        pos_all_feats.clear();
        neg_feats.clear();
    }

    static const int iteration_times = 3;
    std::vector<double> svm_scores(iteration_times, -100);
    int i = 0;
    for (; i < iteration_times; ++i)
    {
        cout << "begin training: round " << i << endl;
        // 1. train with current cache
        string save_fullfile;
        if (save_path)
        {
            save_fullfile = (*save_path + "_train_" + boost::lexical_cast<string>(i));
        }
        svm->SVMTrain(features_cache, labels_cache, string(train_options), save_path ? &save_fullfile:NULL);
        svm_scores[i] = svm->cur_obj();
        cout << "finished training: round " << i << endl;
        return 0;

        // 2. shrink current cache
        {
            cout << "begin shriking current cache: round " << i << endl;
             // 2.1. predict on current cache
            vector<int> predict_labels_cache;
            vector<vector<float>> predict_scores_cache;
            float accuracy_cache;
            string save_fullfile;
            if (save_path)
            {
                save_fullfile = (*save_path + "_predict_cache_" + boost::lexical_cast<string>(i));
            }
            svm->SVMPredict_Primal(features_cache, labels_cache, string(predict_options), &predict_labels_cache, &predict_scores_cache, &accuracy_cache,
                            save_path ? &save_fullfile:NULL);
            // 2.2. identify hard negatives
            std::vector<bool> ishard_vec(bbs_cache.size(), true);
            // IdentifyHardNegativeSamples(features_cache, pos_sample_num, predict_scores_cache, similarity_thresh, &ishard_vec);
            IdentifyHardNegativeSamples(bbs_cache, pos_sample_num, positive_sample_obbs, predict_scores_cache, similarity_thresh, &ishard_vec);

            // 2.3. shrink the current cache: reserve all the positives and hard negatives, easy negatives are abondomed
            std::fill(ishard_vec.begin(), ishard_vec.begin() + pos_sample_num, true);
            GetIndexedSamples(&features_cache, &labels_cache, &predict_scores_cache, &bbs_cache, ishard_vec, -1);
            cout << "finished shriking current cache: round " << i << endl;
            if (features_cache.size() != bbs_cache.size())
            {
                cerr << "features_cache != bbs_cache size" << endl;
                exit(-1);
            }
        }

        // 3. sample new hard negatives and add to cache
        {
            cout << "begin sliding window detect hard negatives: round " << i << endl;
            std::vector<cpu_tsdf::ClassificationResult> res;

            string fullsavepath;
            if (save_path)
            {
                fullsavepath = *save_path + "_sliding_hard_neg_round_" + boost::lexical_cast<string>(i);
            }
//            SlidingBoxDetectionReturningHardNegative(
//                        tsdf_model, template_obb, *svm, 5, 5, M_PI/6,
//                        positive_sample_obbs, pos_features_vecs, bbs_cache, &hard_neg_bbs, &hard_negatives, similarity_thresh,
//                        mesh_min_weight, save_path ? &fullsavepath: NULL);
            //Eigen::Vector3f cur_deltas(0.5, 0.5, 7.5/180.0*M_PI);
            Eigen::Vector3f cur_deltas = detection_deltas;
            const float template_occupy_ratio = 0.05;
            SlidingBoxDetection_Parrellel(tsdf_model,
                                          template_obb, *svm,
                                          template_occupy_ratio,
                                          total_thread,
                                          cur_deltas[0], cur_deltas[1], cur_deltas[2],
                                          x_st_ed, y_st_ed, &res, mesh_min_weight, save_path ? &fullsavepath: NULL);
            std::vector<cpu_tsdf::TSDFFeature> hard_neg_feats;

//            GetHardNegativeSamples(res, tsdf_model, template_feature, pos_all_bbs/positive_sample_obbs, similarity_thresh,
//                                   x_st_ed, y_st_ed, cur_deltas, &hard_neg_feats);
            cout << "begin get hard negatives " << endl;
            GetHardNegativeSamplesParallel(res, tsdf_model, template_feature, positive_sample_obbs, similarity_thresh,
                                           x_st_ed, y_st_ed, cur_deltas, &hard_neg_feats, total_thread);
//            GetHardNegativeSamples(res, tsdf_model, template_feature, positive_sample_obbs, similarity_thresh,
//                                   x_st_ed, y_st_ed, cur_deltas, &hard_neg_feats);
            std::vector<cpu_tsdf::OrientedBoundingBox> hard_neg_bbs_cache;
            std::vector<std::vector<float>> hard_negative_cache;

            AddTSDFFeatureToFeatVec(hard_neg_feats, template_feature, &hard_negative_cache, &hard_neg_bbs_cache, mesh_min_weight);
            std::vector<int> labels(hard_negative_cache.size(), -1);
            int feat_cache_size_before = features_cache.size();
            cout << "feature_cache before: " << features_cache.size() << endl;
            features_cache.insert(features_cache.end(), hard_negative_cache.begin(), hard_negative_cache.end());
            labels_cache.insert(labels_cache.end(), labels.begin(), labels.end());
            bbs_cache.insert(bbs_cache.end(), hard_neg_bbs_cache.begin(), hard_neg_bbs_cache.end());

            RemoveRepeatedNegSamples(pos_sample_num, &features_cache, &labels_cache, &bbs_cache);
            int feat_cache_size_after = features_cache.size();
            cout << "feature_cache after: " << features_cache.size() << endl;
            cout << "hard samples original: " << hard_negative_cache.size() << endl;
            cout << "finished sliding window detect hard negatives: round " << i << endl;
            /////////////////////////////////////////
//            for (int tti = 0; tti < hard_neg_bbs_cache.size(); ++tti)
//            {
//                char temp[128];
//                sprintf(temp, "_hard_neg_%010d_obb_index_%010d_score_%.6f.ply", tti, hard_neg_feats[tti].box_index(), res[hard_neg_feats[tti].box_index()].score);
//                CHECK_EQ(res[hard_neg_feats[tti].box_index()].index, hard_neg_feats[tti].box_index());
//                string cur_save_fname = fullsavepath + temp;
//                WriteOrientedBoundingboxPly(hard_neg_bbs_cache[tti].bb_orientation, hard_neg_bbs_cache[tti].bb_offset,
//                                        hard_neg_bbs_cache[tti].bb_sidelengths, cur_save_fname);

//            }
            /////////////////////////////////////////

            if (features_cache.size() != bbs_cache.size())
            {
                cout << "features_cache != bbs_cache size" << endl;
                exit(-1);
            }
            if (feat_cache_size_before == feat_cache_size_after)
            {
                cout << "feat size break" << endl;
                break;
            }
            if (hard_negative_cache.empty())
            {
                cout << "hard neg empty break" << endl;
                break;
            }
            if (fabs(svm_scores[i] - svm_scores[i-1]) < 1e-4)
            {
                cout << "score no change break" << endl;
                break;
            }
        }       
    }  // end for
    cout << "svm objs: " << endl;
    for (int ti = 0; ti <=i/*svm_scores.size()*/; ++ti)
    {
        cout << ti << ": " << svm_scores[ti] << endl;
    }
}

//int cpu_tsdf::TrainObjectDetector_Old(const cpu_tsdf::TSDFHashing &tsdf_model,
//                                  const cpu_tsdf::OrientedBoundingBox &template_obb,
//                                  const std::vector<cpu_tsdf::OrientedBoundingBox>& positive_sample_obbs,
//                                  int bb_jitter_num_each,
//                                  int random_sample_num,
//                                  SVMWrapper *svm,
//                                  const std::string* save_path,
//                                  float similarity_thresh,
//                                  float mesh_min_weight)
//{
//    using namespace std;
//    const float svm_param_c = 1; //0.01
//    const float svm_param_w1 = 100; //50
//    char train_options[256];
//    sprintf(train_options, "-s 0 -t 0 -c %f -w1 %.9f ", svm_param_c, svm_param_w1);
//    char predict_options[256] = {'\0'};

//    cpu_tsdf::TSDFFeature template_feature;
//    template_feature.ComputeFeature(template_obb.bb_orientation, template_obb.bb_sidelengths, template_obb.bb_offset, template_obb.voxel_lengths, tsdf_model);

//    vector<vector<float>> features_cache;
//    vector<int> labels_cache;
//    vector<cpu_tsdf::OrientedBoundingBox> bbs_cache;
//    int pos_sample_num = 0;
//    std::vector<std::vector<float>> pos_features_vecs;
//    std::vector<cpu_tsdf::OrientedBoundingBox> pos_all_bbs;
//    {
//        // 0. init sample cache
//        // sample positive samples
//        std::vector<cpu_tsdf::TSDFFeature> pos_all_feats;
//        for (int i = 0; i < positive_sample_obbs.size(); ++i)
//        {
//            pos_all_bbs.push_back(positive_sample_obbs[i]);
//            cpu_tsdf::TSDFFeature cur_feat;
//            cur_feat.ComputeFeature(positive_sample_obbs[i].bb_orientation,
//                                    positive_sample_obbs[i].bb_sidelengths,
//                                    positive_sample_obbs[i].bb_offset,
//                                    positive_sample_obbs[i].voxel_lengths,
//                                    tsdf_model);
//            pos_all_feats.push_back(cur_feat);
//        }
//        GenerateSamplesJitteringForMultipleBB(tsdf_model, positive_sample_obbs, bb_jitter_num_each, &pos_all_bbs, &pos_all_feats, save_path, mesh_min_weight);
//        pos_sample_num = pos_all_feats.size();

//        std::vector<cpu_tsdf::TSDFFeature> neg_feats;
//        std::vector<cpu_tsdf::OrientedBoundingBox> neg_bbs;
//        // sample negtive samples
//        string cur_save_path;
//        if (save_path)
//        {
//            cur_save_path = *save_path + "_neg";
//        }
//        RandomGenerateSamples(tsdf_model, template_obb, random_sample_num, template_feature, positive_sample_obbs, &neg_bbs,  &neg_feats, save_path ? &cur_save_path:NULL, mesh_min_weight);

//        // generate obb cache
//        bbs_cache.assign(pos_all_bbs.begin(), pos_all_bbs.end());
//        bbs_cache.insert(bbs_cache.end(), neg_bbs.begin(), neg_bbs.end());
//        // generate feature cache
//        AddTSDFFeatureToFeatVec(pos_all_feats, template_feature, &features_cache, mesh_min_weight);
//        // save positive features, later we compare hard negative features to this to decide real hard negatives
//        pos_features_vecs = features_cache;
//        AddTSDFFeatureToFeatVec(neg_feats, template_feature, &features_cache, mesh_min_weight);
//        labels_cache.assign(pos_all_feats.size() + neg_feats.size(), -1);
//        std::fill(labels_cache.begin(), labels_cache.begin() + pos_all_feats.size(), 1);
//        pos_all_feats.clear();
//        neg_feats.clear();

//        if (features_cache.size() != bbs_cache.size())
//        {
//            cerr << "features_cache != bbs_cache size" << endl;
//            exit(-1);
//        }
//    }

//    for (int i = 0; i < 5; ++i)
//    {
//        cout << "begin training: round " << i << endl;
//        // 1. train with current cache
//        string save_fullfile;
//        if (save_path)
//        {
//            save_fullfile = (*save_path + "_train_" + boost::lexical_cast<string>(i));
//        }
//        svm->SVMTrain(features_cache, labels_cache, string(train_options), save_path ? &save_fullfile:NULL);
//        cout << "finished training: round " << i << endl;

//        // 2. shrink current cache
//        {
//            cout << "begin shriking current cache: round " << i << endl;
//             // 2.1. predict on current cache
//            vector<int> predict_labels_cache;
//            vector<vector<float>> predict_scores_cache;
//            float accuracy_cache;
//            string save_fullfile;
//            if (save_path)
//            {
//                save_fullfile = (*save_path + "_predict_cache_" + boost::lexical_cast<string>(i));
//            }
//            svm->SVMPredict(features_cache, labels_cache, string(predict_options), &predict_labels_cache, &predict_scores_cache, &accuracy_cache,
//                            save_path ? &save_fullfile:NULL);
//            // 2.2. identify hard negatives
//            std::vector<bool> ishard_vec;
//            GetHardNegativeSamples(features_cache, pos_sample_num, predict_scores_cache, similarity_thresh, &ishard_vec);
//            // 2.3. shrink the current cache: reserve all the positives and hard negatives, easy negatives are abondomed
//            std::fill(ishard_vec.begin(), ishard_vec.begin() + pos_sample_num, true);
//            GetIndexedSamples(&features_cache, &labels_cache, &predict_scores_cache, &bbs_cache, ishard_vec, -1);
//            cout << "finished shriking current cache: round " << i << endl;
//            if (features_cache.size() != bbs_cache.size())
//            {
//                cerr << "features_cache != bbs_cache size" << endl;
//                exit(-1);
//            }
//        }

//        // 3. sample new hard negatives and add to cache
//        {
//            cout << "begin sliding window detect hard negatives: round " << i << endl;
//            std::vector<cpu_tsdf::OrientedBoundingBox> hard_neg_bbs;
//            std::vector<std::vector<float>> hard_negatives;
//            string fullsavepath;
//            if (save_path)
//            {
//                fullsavepath = *save_path + "_sliding_hard_neg_round_" + boost::lexical_cast<string>(i);
//            }
//            SlidingBoxDetectionReturningHardNegative(
//                        tsdf_model, template_obb, *svm, 5, 5, M_PI/6,
//                        positive_sample_obbs, pos_features_vecs, bbs_cache, &hard_neg_bbs, &hard_negatives, similarity_thresh,
//                        mesh_min_weight, save_path ? &fullsavepath: NULL);

//            std::vector<int> labels(hard_negatives.size(), -1);
//            features_cache.insert(features_cache.end(), hard_negatives.begin(), hard_negatives.end());
//            labels_cache.insert(labels_cache.end(), labels.begin(), labels.end());
//            bbs_cache.insert(bbs_cache.end(), hard_neg_bbs.begin(), hard_neg_bbs.end());
//            cout << "hard samples: " << hard_negatives.size() << endl;
//            cout << "finished sliding window detect hard negatives: round " << i << endl;
//            if (features_cache.size() != bbs_cache.size())
//            {
//                cerr << "features_cache != bbs_cache size" << endl;
//                exit(-1);
//            }
//            if (hard_negatives.empty())
//            {
//                break;
//            }
//        }
//        ///////////////////////////////////////////////////////////
//    }  // end for
//}


void cpu_tsdf::GenerateSamplesJitteringForMultipleBB(
        const cpu_tsdf::TSDFHashing &tsdf_origin,
        const std::vector<cpu_tsdf::OrientedBoundingBox> &sample_bbs,
        int bb_num_each,
        std::vector<cpu_tsdf::TSDFFeature> *features,
        const std::string *save_path, float mesh_min_weight)
{

    cpu_tsdf::WriteOrientedBoundingboxesPly(sample_bbs, *save_path + "template_debug_obb.ply");
    for (int i = 0; i < sample_bbs.size(); ++i)
    {
        const cpu_tsdf::OrientedBoundingBox& template_bb = sample_bbs[i];
        cpu_tsdf::TSDFFeature template_feature;
        template_feature.ComputeFeature(template_bb, tsdf_origin);

        std::unique_ptr<std::string> cur_path_p;
        if (save_path)
        {
            char tmp[20];
            sprintf(tmp, "%05d", i);
            cur_path_p.reset(new std::string(*save_path + "_posnum_" + tmp));
        }
        GenerateSamplesJittering(tsdf_origin, template_feature, bb_num_each, features, cur_path_p.get(), mesh_min_weight);
    }
}


int cpu_tsdf::AddTSDFFeatureToFeatVec(const std::vector<cpu_tsdf::TSDFFeature> tsdf_feats,
                                      const cpu_tsdf::TSDFFeature& template_feat,
                                      std::vector<std::vector<float> > *feat_vec,
                                      std::vector<cpu_tsdf::OrientedBoundingBox>* bbs_vec,
                                      const float mesh_min_weight)
{
    for (int i = 0; i < tsdf_feats.size(); ++i)
    {
        std::vector<float> converted_feat;
        tsdf_feats[i].GetFeatureVector(&template_feat, &converted_feat, mesh_min_weight, true);
        feat_vec->push_back(converted_feat);
        bbs_vec->push_back(tsdf_feats[i].GetOrientedBoundingBox());
    }
    return 0;
}


std::vector<float> cpu_tsdf::NormalizeVector(const std::vector<float> vec)
{
    float sum = 0;
    for (int i = 0; i < vec.size(); ++i)
    {
        sum += (vec[i] * vec[i]);
    }
    sum = std::sqrt(sum);
    std::vector<float> res(vec.size());
    for (int i =0; i < vec.size(); ++i)
    {
        res[i] = vec[i]/sum;
    }
    return res;
}


bool cpu_tsdf::TestIsVectorSimilar(const std::vector<std::vector<float> > &normed_vecs, const std::vector<float> &test_vec, const float thresh)
{
    std::vector<float> normed_cur_vec = NormalizeVector(test_vec);
    for (int i = 0; i < normed_vecs.size(); ++i)
    {
        if (std::inner_product(normed_vecs[i].begin(), normed_vecs[i].end(), normed_cur_vec.begin(), 0.0)
                > thresh)
        {
            return true;
        }
    }
    return false;
}


//int cpu_tsdf::IdentifyHardNegativeSamples_Old(const std::vector<std::vector<float> > &tsdf_feats, const int gt_pos_num, const std::vector<std::vector<float> > &predict_scores, const float thresh, std::vector<bool> *ishard_vec)
//{
//    std::vector<std::vector<float>> normed_tsdf_feats(gt_pos_num);
//    for (int i = 0; i < gt_pos_num; ++i)
//    {
//        normed_tsdf_feats[i] = NormalizeVector(tsdf_feats[i]);
//    }

//    ishard_vec->assign(tsdf_feats.size(), false);
//    for (int i = gt_pos_num; i < tsdf_feats.size(); ++i)
//    {
//        int cur_label = -1;
//        if ( cur_label * predict_scores[i][0] < 1 /* within margin or wrong*/
//             && !TestIsVectorSimilar(normed_tsdf_feats, tsdf_feats[i], thresh) /* not similar with any pos sample */
//             )
//        {
//            /*hard*/
//            (*ishard_vec)[i] = true;
//        }
//    }
//}

bool cpu_tsdf::TestOBBSimilarity(
        const float similarity_thresh,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const cpu_tsdf::OrientedBoundingBox& cur_bb, int& cur_intersect_pos_sample, float& cur_intersect_area)
{
//    int cur_intersect_pos_sample;
//    float cur_intersect_area;
    bool is_intersect = OBBsLargeOverlapArea(cur_bb, pos_bbs, &cur_intersect_pos_sample, &cur_intersect_area, 0.1);
    float cos_similarity = cur_bb.bb_orientation.col(0).dot(pos_bbs[cur_intersect_pos_sample].bb_orientation.col(0));
    if (is_intersect && cur_intersect_area * std::max(cos_similarity, 0.0f) > similarity_thresh) /* a real positive sample that should have high score*/
    {
        return true;
    }
    return false;
}

int cpu_tsdf::GetHardNegativeSamplesParallel_thread(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats,
        int start,
        int end
        )
{
    const cpu_tsdf::OrientedBoundingBox& template_bb = template_feature.GetOrientedBoundingBox();
    const Eigen::Vector3f template_center = template_bb.BoxCenter();

    cout << "parallel start: " << start << endl;
    cout << "parallel end: " << end << endl;
    CHECK_LE(end, res.size());
    for (int i = start; i < end; ++i)
    {
        // non-empty
        const int cur_box_idx = res[i].index;
        if (cur_box_idx == -1)
        {
            continue;
        }
        CHECK_EQ(i, res[i].index);
        // ensure it's "hard"
        const int cur_label = -1;
        if ( cur_label * res[i].score >= 1) /* not within margin nor wrong, not hard negative*/
        {
            continue;
        }
        // ensure it's negative
        // get bounding box

        float cur_x, cur_y, cur_angle;
        BoxIndexToBoxPos(x_st_ed, y_st_ed, deltas[0], deltas[1], deltas[2], cur_box_idx, &cur_x, &cur_y, &cur_angle);
        Eigen::Vector3f cur_center(cur_x,  cur_y,  template_center[2]);
        cpu_tsdf::OrientedBoundingBox cur_bb;
        GenerateBoundingbox(template_bb, cur_center, cur_angle, &cur_bb);
        int cur_intersect_pos_sample;
        float cur_intersect_area;
        bool is_intersect = OBBsLargeOverlapArea(cur_bb, pos_bbs, &cur_intersect_pos_sample, &cur_intersect_area, 0.1);
        float cos_similarity = cur_bb.bb_orientation.col(0).dot(pos_bbs[cur_intersect_pos_sample].bb_orientation.col(0));
        if (is_intersect && cur_intersect_area * std::max(cos_similarity, 0.0f) > similarity_thresh) /* a real positive sample that should have high score*/
        {
            continue;
        }
        //TestOBBSimilarity(similarity_thresh, pos_bbs, cur_bb);
        /*hard negative*/
        cpu_tsdf::TSDFFeature new_hard_neg(cur_bb, tsdf_model);
        new_hard_neg.box_index(cur_box_idx);
        hard_neg_feats->push_back(new_hard_neg);
        // std::cout << "hard neg: " << cur_box_idx << std::endl;
    }
}

int cpu_tsdf::GetHardNegativeSamplesParallel(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats,
        const int thread_num
        )
{
    using namespace boost::threadpool;
    std::vector<std::vector<cpu_tsdf::TSDFFeature>> parall_hard_neg_feats(thread_num);
    //std::vector<std::string> save_paths(thread_num);

    {
        int total_thread = thread_num;
        // Create a thread pool.
        pool tp(thread_num);
        // Add some tasks to the pool.
        int seg_length = round(double(res.size())/double(total_thread));
        for (int cur_thread = 0; cur_thread < total_thread; ++cur_thread)
        {
            int start = 0 + seg_length * cur_thread;
            int end = std::min<int> (0 + seg_length * (cur_thread + 1), res.size());
//            GetHardNegativeSamplesParallel_thread(
//                        res, tsdf_model, template_feature, positive_sample_obbs, similarity_thresh,
//                                   x_st_ed, y_st_ed, cur_deltas, &parall_hard_neg_feats[cur_thread], start, end);
            tp.schedule(std::bind(&GetHardNegativeSamplesParallel_thread, std::cref(res), std::cref(tsdf_model),
                                  std::cref(template_feature), std::cref(pos_bbs), std::cref(similarity_thresh),
                                  x_st_ed, y_st_ed, std::cref(deltas), &parall_hard_neg_feats[cur_thread], start, end));

        }
    }
    // combine the results into one vector
    cout << "combining hard negatives" << endl;
    CHECK_EQ(thread_num, parall_hard_neg_feats.size());
    for (int k = 0; k < thread_num; ++k)
    {
        cout << k << "th parall hard" << endl;
        hard_neg_feats->insert(hard_neg_feats->end(), parall_hard_neg_feats[k].begin(), parall_hard_neg_feats[k].end());
        parall_hard_neg_feats[k].clear();
    }
    return 1;
}

int cpu_tsdf::GetHardNegativeSamples(
        const std::vector<cpu_tsdf::ClassificationResult>& res,
        const cpu_tsdf::TSDFHashing& tsdf_model,
        const cpu_tsdf::TSDFFeature& template_feature,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const float similarity_thresh,
        const float* x_st_ed, const float* y_st_ed, const Eigen::Vector3f& deltas,
        std::vector<cpu_tsdf::TSDFFeature>* hard_neg_feats
        )
{
    const cpu_tsdf::OrientedBoundingBox& template_bb = template_feature.GetOrientedBoundingBox();
    const Eigen::Vector3f template_center = template_bb.BoxCenter();

    for (int i = 0; i < res.size(); ++i)
    {
        // non-empty
        const int cur_box_idx = res[i].index;
        if (cur_box_idx == -1)
        {
            continue;
        }
        CHECK_EQ(i, res[i].index);
        // ensure it's "hard"
        const int cur_label = -1;
        if ( cur_label * res[i].score >= 1) /* not within margin nor wrong, not hard negative*/
        {
            continue;
        }
        // ensure it's negative
        // get bounding box

        float cur_x, cur_y, cur_angle;
        BoxIndexToBoxPos(x_st_ed, y_st_ed, deltas[0], deltas[1], deltas[2], cur_box_idx, &cur_x, &cur_y, &cur_angle);
        Eigen::Vector3f cur_center(cur_x,  cur_y,  template_center[2]);
        cpu_tsdf::OrientedBoundingBox cur_bb;
        GenerateBoundingbox(template_bb, cur_center, cur_angle, &cur_bb);
        int cur_intersect_pos_sample;
        float cur_intersect_area;
        bool is_intersect = OBBsLargeOverlapArea(cur_bb, pos_bbs, &cur_intersect_pos_sample, &cur_intersect_area, 0.1);
        float cos_similarity = cur_bb.bb_orientation.col(0).dot(pos_bbs[cur_intersect_pos_sample].bb_orientation.col(0));
        if (is_intersect && cur_intersect_area * std::max(cos_similarity, 0.0f) > similarity_thresh) /* a real positive sample that should have high score*/
        {
            continue;
        }
        /*hard negative*/
        cpu_tsdf::TSDFFeature new_hard_neg(cur_bb, tsdf_model);
        new_hard_neg.box_index(cur_box_idx);
        hard_neg_feats->push_back(new_hard_neg);
        // std::cout << "hard neg: " << cur_box_idx << std::endl;
    }
}

int cpu_tsdf::IdentifyHardNegativeSamples(
        const std::vector<cpu_tsdf::OrientedBoundingBox>& all_obbs,
        const int gt_pos_num,
        const std::vector<cpu_tsdf::OrientedBoundingBox>& pos_bbs,
        const std::vector<std::vector<float> > &predict_scores,
        const float similarity_thresh,
        std::vector<bool> *ishard_vec)
{
//    std::vector<cpu_tsdf::OrientedBoundingBox> pos_bbs(all_obbs.begin(), all_obbs.begin() + gt_pos_num);
    ishard_vec->assign(all_obbs.size(), false);
    for (int i = gt_pos_num; i < all_obbs.size(); ++i)
    {
        const int cur_label = -1;
        if ( cur_label * predict_scores[i][0] >= 1) /* not within margin nor wrong, not hard negative*/
        {
            continue;
        }
        const cpu_tsdf::OrientedBoundingBox& cur_bb = all_obbs[i];
        int cur_intersect_pos_sample;
        float cur_intersect_area;
        bool is_intersect = OBBsLargeOverlapArea(cur_bb, pos_bbs, &cur_intersect_pos_sample, &cur_intersect_area, 0.1);
        float cos_similarity = cur_bb.bb_orientation.col(0).dot(pos_bbs[cur_intersect_pos_sample].bb_orientation.col(0));
        if (is_intersect && cur_intersect_area * std::max(cos_similarity, 0.0f) > similarity_thresh) /* a real positive sample that should have high score*/
        {
            continue;
        }
        /*hard*/
        (*ishard_vec)[i] = true;
    }
    return 0;
}


int cpu_tsdf::GetIndexedSamples(std::vector<std::vector<float> > *tsdf_feats,
                                std::vector<int> *labels,
                                std::vector<std::vector<float> > *predict_scores,
                                std::vector<cpu_tsdf::OrientedBoundingBox>* bbs,
                                const std::vector<bool> &index_vec, const int reserved_number)
{
    std::vector<std::vector<float>> ntsdf_feats;
    std::vector<int> nlabels;
    std::vector<std::vector<float>> npredict_scores;
    std::vector<cpu_tsdf::OrientedBoundingBox> nbbs;
    assert(index_vec.size() == tsdf_feats->size());
    int cur_reserved_num = reserved_number < 0 ? index_vec.size():reserved_number;
    int cnt = 0;
    for (int i = 0; i < index_vec.size(); ++i)
    {
        if (index_vec[i])
        {
            ntsdf_feats.push_back((*tsdf_feats)[i]);
            nlabels.push_back((*labels)[i]);
            npredict_scores.push_back((*predict_scores)[i]);
            nbbs.push_back((*bbs)[i]);
            cnt ++ ;
            if (cnt >= cur_reserved_num)
            {
                break;
            }
        } // end if
    } // end for
    *tsdf_feats = ntsdf_feats;
    *labels = nlabels;
    *predict_scores = npredict_scores;
    *bbs = nbbs;
}

void cpu_tsdf::SlidingBoxDetection_OneRotationAngle2(const cpu_tsdf::TSDFHashing &tsdf_model,
                                                    const cpu_tsdf::OrientedBoundingBox& template_bb,
                                                    const float template_occpy_ratio,
                                                    const SVMWrapper &svm,
                                                    const float delta_x, const float delta_y, const float delta_rotation, const float rotate_angle,
                                                    const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt,
                                                    std::vector<cpu_tsdf::ClassificationResult> * res, const float mesh_min_weight, const std::string *save_path)
{
    try {
    FILE* hf = NULL;
    if (save_path)
    {
        hf = fopen(((*save_path) + "_SlidingBoxDetectionResults.txt").c_str(), "w");
        fprintf(hf, "index label score\n");
    }

    using namespace std;
    Eigen::Vector3f box_center = template_bb.BoxCenter();
//    cpu_tsdf::TSDFFeature template_feature;
//    template_feature.ComputeFeature(template_bb, tsdf_model);
//    const float template_occpy_ratio = template_feature.occupied_ratio(mesh_min_weight);

    float x_st_ed[2];
    float y_st_ed[2];
    x_st_ed[0] = min_pt[0];
    x_st_ed[1] = max_pt[0];

    y_st_ed[0] = min_pt[1];
    y_st_ed[1] = max_pt[1];

    res->clear();
    cout << "x sample num: " << floor((max_pt[0] - min_pt[0])/delta_x) + 1  << endl;
    cout << "y sample num: " << floor((max_pt[1] - min_pt[1])/delta_y) + 1  << endl;
    cout << "rotate sample num: " << floor((2*M_PI - 1e-5)/delta_rotation) + 1 << endl;
    int x_s_num = floor((max_pt[0] - min_pt[0])/delta_x) + 1;
    int y_s_num = floor((max_pt[1] - min_pt[1])/delta_y) + 1;
    int rotate_s_num = 1;
    int original_rotate_s_num = floor((2*M_PI - 1e-5)/delta_rotation) + 1;
    fprintf(hf, "min_pt: %f %f %f\n", min_pt[0], min_pt[1], min_pt[2]);
    fprintf(hf, "max_pt: %f %f %f\n", max_pt[0], max_pt[1], max_pt[2]);
    fprintf(hf, "delta_x, delta_y, delta_r: %f %f %f\n", delta_x, delta_y, delta_rotation);

    int init_box_index = -1;
    BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, min_pt[0], min_pt[1], rotate_angle, &init_box_index);
    int delta_box_index = original_rotate_s_num;
    int box_index = init_box_index - delta_box_index;

    /////////////////////////////////////////////////////////////////////////////
    // test new detector and feature extraction
    {
        cerr << "start new code" << endl;
        using namespace Eigen;
        tsdf_detection::SceneDiscretizeInfo scene_info(Vector2f(x_st_ed[0], x_st_ed[1]),
                Vector2f(y_st_ed[0], y_st_ed[1]),
                Vector3f(delta_x, delta_y, delta_rotation));
        tsdf_detection::DetectionParams params;
        params.do_NMS = false;
        params.min_score_to_keep = 0.0;
        params.save_prefix = "/home/dell/newcodedebug/";
        Eigen::Vector3f sample_sizef = template_bb.bb_sidelengths.cwiseQuotient((template_bb.voxel_lengths)) + Eigen::Vector3f::Ones();
        Eigen::Vector3i sample_size(round(sample_sizef[0]), round(sample_sizef[1]), round(sample_sizef[2]));
        tsdf_detection::Detector detector(svm.model_path_);
        cerr << svm.model_path_ << endl;
        tsdf_detection::SampleTemplate sample_template(tsdf_utility::NewOBBFromOld(template_bb), sample_size);
        float angle = acos(template_bb.bb_orientation(0, 0));
        if (template_bb.bb_orientation(1, 0) < 0) {
            angle *= -1;
        }
        if (angle < 0) {
            angle += M_PI * 2;
        }
        tsdf_detection::SampleCollection samples;
        int disc_angle = round(rotate_angle/delta_rotation);
        cerr <<disc_angle<<endl;
        tsdf_detection::Detect_OneAngle(tsdf_model, detector, scene_info, params, disc_angle, angle, &samples);
        samples.WriteOBBsToPLY(params.save_prefix + "newcode.ply");
       // // tsdf_utility::OrientedBoundingBox cur_newobb(cur_bb.bb_sidelengths, cur_bb.BoxBottomCenter(), angle);
       // tsdf_utility::OrientedBoundingBox cur_newobb = tsdf_utility::ComputeNewOBBFromOld(cur_bb);
       // tsdf_detection::Sample sample(cur_newobb, sample_size, tsdf_model, scene_info.OBBPos2Index(Vector3f(curx, cury, angle)), mesh_min_weight);
       // float newobb_score;
       // char label;
       // cerr << "newcode: sample_size: " <<  sample_size << endl;
       // cerr << "newcode: angle: " << angle << "\t original: fangle: " << rotate_angle << endl;
       // cerr << "newcode: newobb:\n " << cur_newobb.AffineTransform().matrix() << endl;
       // cerr << "newcode: newobb offset:\n " << cur_newobb.Offset() << endl;
       // cerr << "oldcode: oldlbb offset:\n " << cur_bb.BoxBottomCenter() << endl;
       // Eigen::Affine3f oldaffine;
       // cpu_tsdf::OBBToAffine(cur_bb, &(oldaffine));
       // cout << "old code: oldobb: \n" <<  oldaffine.matrix() << endl;
       // sample.OutputToText("/home/dell/test1.txt", 0);
       // //cerr << "start new code2" << endl;
       // detector.Predict(params, &sample, &newobb_score, &label);
       // cout << "newcode: curscore: " << newobb_score << endl;
       // cout << "xi, yi: " << xi << " " << yi << endl;
       // new_obb_score_outscope = newobb_score;
        cerr << "end new code" << endl;
    }
    /////////////////////////////////////////////////////////////////////////////

    res->resize(x_s_num * y_s_num * rotate_s_num);
//    for (float curx = min_pt[0]; curx < max_pt[0]; curx += delta_x)
//        for (float cury = min_pt[1]; cury < max_pt[1]; cury += delta_y)
    for (int xi = 0; xi < x_s_num; ++xi)
        for (int yi = 0; yi < y_s_num; ++yi)
    //for (int xi = 14; xi < x_s_num; ++xi)
    //    for (int yi = 40; yi < y_s_num; ++yi)
            //for (float rotate_angle = 0; rotate_angle < 2 * M_PI - 1e-5; rotate_angle += delta_rotation)
            {
                float curx = min_pt[0] + xi * delta_x;
                float cury = min_pt[1] + yi * delta_y;
                box_index += delta_box_index;
                //cout << "box: " << box_index << endl;

//                float scale_max_score = -1e6;
//                for (float scalex = 1; scalex < 1.2 + 1e-3; scalex += 0.05)
//                    for (float scaley = 1; scaley < 1.2 + 1e-3; scaley += 0.05)
//                        for (float scalez = 1; scalez < 1.2 + 1e-3; scalez += 0.05)
                cpu_tsdf::OrientedBoundingBox cur_bb;
                Eigen::Vector3f cur_center = Eigen::Vector3f(curx, cury, box_center[2]);
                GenerateBoundingbox(template_bb, cur_center, rotate_angle, &cur_bb);
                 {
                     //cout << "xi: " << xi << " yi: " << yi << "ri: " << (rotate_angle/delta_rotation) << endl;
                    float tposx, tposy, tangle;
                    BoxIndexToBoxPos(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, box_index, &tposx, &tposy, &tangle);
                    CHECK_LT(fabs(tposx - cur_bb.BoxCenter()[0]), 1e-3);
                    CHECK_LT(fabs(tposy - cur_bb.BoxCenter()[1]), 1e-3);
                    CHECK_LT(fabs(tangle - rotate_angle), 1e-3);

                    int cur_idx_test = -1;
                    cv::Vec3f test_axis_angle;
                    Eigen::Matrix3f tmp = (cur_bb.bb_orientation * template_bb.bb_orientation.transpose().eval());
                    cv::Rodrigues(utility::EigenMatToCvMatx(tmp), test_axis_angle);
                    double fangle = test_axis_angle[2];
                    if (fangle < 0)
                    {
                        fangle += 2 * M_PI;
//                        cout << fangle << endl;
//                        cout << rotate_angle << endl;
                    }
                    CHECK_LT(fabs(fangle - rotate_angle), 1e-3);
                    CHECK_LT(fabs(cur_bb.BoxCenter()[0] - curx), 1e-3);
                    CHECK_LT(fabs(cur_bb.BoxCenter()[1] - cury), 1e-3);
                    CHECK_GE(fangle, 0);
                    bool res = BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, cur_bb.BoxCenter()[0], cur_bb.BoxCenter()[1], fangle, &cur_idx_test);
                    CHECK_EQ(cur_idx_test, box_index);
                    CHECK_EQ(res, true);

                }

//                if (save_path)
//                {
//                    const std::string& output_dir_prefix = *save_path;
//                    char number[10];
//                    sprintf(number, "_%05d", box_index);
//                    string obb_name = output_dir_prefix + "_" + number + "_feat_debug_all_obb.ply";
//                    cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);
//                }

                cpu_tsdf::TSDFFeature cur_feat;
                if (cpu_tsdf::GenerateOneSample(tsdf_model, cur_bb, template_occpy_ratio, mesh_min_weight, &cur_feat))
                {
                    /////////////////////////////////////////////////////////////////////
                    // test new detector and feature extraction
                    //float new_obb_score_outscope = 0;
                    //{
                    //    cerr << "start new code" << endl;
                    //    using namespace Eigen;
                    //    tsdf_detection::SceneDetectInfo scene_info(Vector2f(x_st_ed[0], x_st_ed[1]),
                    //            Vector2f(y_st_ed[0], y_st_ed[1]),
                    //            Vector3f(delta_x, delta_y, delta_rotation));
                    //    tsdf_detection::DetectionParams params;
                    //    Eigen::Vector3f sample_sizef = template_bb.bb_sidelengths.cwiseQuotient((template_bb.voxel_lengths)) + Eigen::Vector3f::Ones();
                    //    Eigen::Vector3i sample_size(round(sample_sizef[0]), round(sample_sizef[1]), round(sample_sizef[2]));
                    //    tsdf_detection::Detector detector(svm.model_path_, template_bb.bb_sidelengths, sample_size, template_bb.BoxBottomCenter()[2], template_occpy_ratio);
                    //    float angle = acos(cur_bb.bb_orientation(0, 0));
                    //    if (cur_bb.bb_orientation(1, 0) < 0) {
                    //        angle *= -1;
                    //    }
                    //    if (angle < 0) {
                    //        angle += M_PI * 2;
                    //    }
                    //    // tsdf_utility::OrientedBoundingBox cur_newobb(cur_bb.bb_sidelengths, cur_bb.BoxBottomCenter(), angle);
                    //    tsdf_utility::OrientedBoundingBox cur_newobb = tsdf_utility::ComputeNewOBBFromOld(cur_bb);
                    //    tsdf_detection::Sample sample(cur_newobb, sample_size, tsdf_model, scene_info.OBBPos2Index(Vector3f(curx, cury, angle)), mesh_min_weight);
                    //    float newobb_score;
                    //    char label;
                    //    cerr << "newcode: sample_size: " <<  sample_size << endl;
                    //    cerr << "newcode: angle: " << angle << "\t original: fangle: " << rotate_angle << endl;
                    //    cerr << "newcode: newobb:\n " << cur_newobb.AffineTransform().matrix() << endl;
                    //    cerr << "newcode: newobb offset:\n " << cur_newobb.Offset() << endl;
                    //    cerr << "oldcode: oldlbb offset:\n " << cur_bb.BoxBottomCenter() << endl;
                    //    Eigen::Affine3f oldaffine;
                    //    cpu_tsdf::OBBToAffine(cur_bb, &(oldaffine));
                    //    cout << "old code: oldobb: \n" <<  oldaffine.matrix() << endl;
                    //    sample.OutputToText("/home/dell/test1.txt", 0);
                    //    //cerr << "start new code2" << endl;
                    //    detector.Predict(params, &sample, &newobb_score, &label);
                    //    cout << "newcode: curscore: " << newobb_score << endl;
                    //    cout << "xi, yi: " << xi << " " << yi << endl;
                    //    new_obb_score_outscope = newobb_score;
                    //}
                    /////////////////////////////////////////////////////////////////////
                    vector<vector<float>> feat_vec(1);
                    cur_feat.GetFeatureVector(/*&template_feature*/NULL,  &(feat_vec[0]), mesh_min_weight, true);
                    cur_feat.OutputToText2("/home/dell/test2.2.txt", 0);
                    vector<int> input_labels(1, -1);
                    vector<int> output_label;
                    vector<vector<float>> output_score;
                    float accuracy;
                    //svm.SVMPredict(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
                    svm.SVMPredict_Primal(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
                    int cur_label = output_label[0];
                    float cur_score = output_score[0][0];
                        // cerr << "oldcode: oldscore: " << cur_score<< endl;
//                    fprintf(hf, "pos: %f %f, angle: %f\n, feat_size: %d\n", curx, cury, rotate_angle, feat_vec[0].size());
//                    for (int tti = 0; tti < feat_vec[0].size(); ++tti)
//                    {
//                        fprintf(hf, "%d:%f,  ", tti, feat_vec[0][tti]);
//                    }
//                    fprintf(hf, "\n");
                    fprintf(hf, "%d %d %f\n", box_index, cur_label, cur_score);
                    // CHECK_LT(fabs(cur_score - new_obb_score_outscope), 0.1);
                    //char ch;
                    //cin >> ch;
                    //fflush(hf);
                    ////////////////////////////

                    if ( cur_score >= 0.0)
                    {
                        //output the hard negative samples
                        if (save_path)
                        {
                            /********/
                            cout << "save " << box_index << "th bb" << endl;
                            char weight_num[10];
                            sprintf(weight_num, "%.02f", mesh_min_weight);
                            if (save_path)
                            {
                                using namespace std;
                                char cscore[15];
                                char clabel[5];
                                sprintf(cscore, "%.04f", cur_score);
                                sprintf(clabel, "%02d", cur_label);
                                const std::string output_dir_prefix = *save_path + "_detected_label_" + clabel + "_score_" + cscore;
                                char number[10];
                                sprintf(number, "_%05d", box_index);
                                string feat_name = output_dir_prefix + "_boxidx_" + number + "_feat_" + weight_num + ".txt";
                                vector<float> neg_out_feat;
//                                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                                utility::OutputVector(feat_name, neg_out_feat);

//                                string feat_name = output_dir_prefix + "_boxidx_" + number + "_obb.txt";
//                               cpu_tsdf::WriteOrientedBoundingboxPly()
                                //assert(neg_out_feat.size() == out_feat_template.size());

//                                cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                                SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_NN.bin";
//                                std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                boost::archive::binary_oarchive oa(os);
//                                oa << *(cur_feat_tsdf);
//                                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + "_NN.ply";
//                                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);

//                                {
//                                    cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                                    SliceTSDFWithBoundingbox(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                                    string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_Linear.bin";
//                                    std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                    boost::archive::binary_oarchive oa(os);
//                                    oa << *(cur_feat_tsdf);
//                                    string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + "_linear.ply";
//                                    cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
//                                }

                                string obb_name = output_dir_prefix + "_" + number + "_feat_obb.ply";
                                cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);




                            }  //
                            cout << "save " << box_index << "th bb finished" << endl;
                            /*******/
                        }
                    }
                    ////////////////////////////
                    //std::cout << "hard neg box idx: " << box_index << std::endl;
                    (*res)[box_index/delta_box_index] = (ClassificationResult(box_index, cur_label, cur_score));
                }
            }
    if (save_path)
    {
        fclose(hf);
    }
    }
    catch(std::exception & e)
    {
        cout << "exception in thread" << endl;
        cout << e.what() << endl;
    }
    catch (...)
    {
        cout << "unknown exception in thread" << endl;
    }

    return;

}


void cpu_tsdf::SlidingBoxDetection_OneRotationAngle(const cpu_tsdf::TSDFHashing &tsdf_model,
                                                    const cpu_tsdf::OrientedBoundingBox &template_bb,
                                                    const SVMWrapper &svm,
                                                    const float delta_x, const float delta_y, const float delta_rotation, const float rotate_angle,
                                                    const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt,
                                                    std::vector<cpu_tsdf::ClassificationResult> * res, const float mesh_min_weight, const std::string *save_path)
{
    try {
    FILE* hf = NULL;
    if (save_path)
    {
        hf = fopen(((*save_path) + "_SlidingBoxDetectionResults.txt").c_str(), "w");
        fprintf(hf, "index label score\n");
    }

    using namespace std;
//    Eigen::Vector3f min_pt, max_pt;
//    tsdf_model.getBoundingBoxInWorldCoord(min_pt, max_pt);
    Eigen::Vector3f box_center = template_bb.BoxCenter();
    cpu_tsdf::TSDFFeature template_feature;
    template_feature.ComputeFeature(template_bb, tsdf_model);
    const float template_occpy_ratio = template_feature.occupied_ratio(mesh_min_weight);

    float x_st_ed[2];
    float y_st_ed[2];
    x_st_ed[0] = min_pt[0];
    x_st_ed[1] = max_pt[0];

    y_st_ed[0] = min_pt[1];
    y_st_ed[1] = max_pt[1];

    res->clear();
    cout << "x sample num: " << floor((max_pt[0] - min_pt[0])/delta_x) + 1  << endl;
    cout << "y sample num: " << floor((max_pt[1] - min_pt[1])/delta_y) + 1  << endl;
    cout << "rotate sample num: " << floor((2*M_PI - 1e-5)/delta_rotation) + 1 << endl;
    int x_s_num = floor((max_pt[0] - min_pt[0])/delta_x) + 1;
    int y_s_num = floor((max_pt[1] - min_pt[1])/delta_y) + 1;
    int rotate_s_num = 1;
    int original_rotate_s_num = floor((2*M_PI - 1e-5)/delta_rotation) + 1;
    fprintf(hf, "min_pt: %f %f %f\n", min_pt[0], min_pt[1], min_pt[2]);
    fprintf(hf, "max_pt: %f %f %f\n", max_pt[0], max_pt[1], max_pt[2]);
    fprintf(hf, "delta_x, delta_y, delta_r: %f %f %f\n", delta_x, delta_y, delta_rotation);

    int init_box_index = -1;
    BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, min_pt[0], min_pt[1], rotate_angle, &init_box_index);
    int delta_box_index = original_rotate_s_num;
    int box_index = init_box_index - delta_box_index;

    res->resize(x_s_num * y_s_num * rotate_s_num);
//    for (float curx = min_pt[0]; curx < max_pt[0]; curx += delta_x)
//        for (float cury = min_pt[1]; cury < max_pt[1]; cury += delta_y)
    for (int xi = 0; xi < x_s_num; ++xi)
        for (int yi = 0; yi < y_s_num; ++yi)
            //for (float rotate_angle = 0; rotate_angle < 2 * M_PI - 1e-5; rotate_angle += delta_rotation)
            {
                float curx = min_pt[0] + xi * delta_x;
                float cury = min_pt[1] + yi * delta_y;
                box_index += delta_box_index;
                //box_index++;
                //cout << "box: " << box_index << endl;
///debug
//                float debugx, debugy, debugang;
//                BoxIndexToBoxPos(x_st_ed,  y_st_ed,
//                                      delta_x, delta_y, delta_rotation,
//                                      box_index, &debugx,  &debugy, &debugang);
//                assert(fabs(debugx - curx) < 1e-3 && fabs(debugy - cury) < 1e-3 && fabs(debugang - rotate) < 1e-3);
//                int box_ind;
//                BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, curx, cury, rotate, &box_ind);
//                assert(box_ind == box_index);

                cpu_tsdf::OrientedBoundingBox cur_bb;
                Eigen::Vector3f cur_center = Eigen::Vector3f(curx, cury, box_center[2]);
                GenerateBoundingbox(template_bb, cur_center, rotate_angle, &cur_bb);

                //BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, min_pt[0], min_pt[1], rotate_angle, &init_box_index);
                {
                    // cout << "xi: " << xi << " yi: " << yi << "ri: " << (rotate_angle/delta_rotation) << endl;
                    float tposx, tposy, tangle;
                    BoxIndexToBoxPos(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, box_index, &tposx, &tposy, &tangle);
                    CHECK_LT(fabs(tposx - cur_bb.BoxCenter()[0]), 1e-3);
                    CHECK_LT(fabs(tposy - cur_bb.BoxCenter()[1]), 1e-3);
                    CHECK_LT(fabs(tangle - rotate_angle), 1e-3);

                    int cur_idx_test = -1;
                    cv::Vec3f test_axis_angle;
                    Eigen::Matrix3f tmp = (cur_bb.bb_orientation * template_bb.bb_orientation.transpose().eval());
                    cv::Rodrigues(utility::EigenMatToCvMatx(tmp), test_axis_angle);
                    double fangle = test_axis_angle[2];
                    if (fangle < 0)
                    {
                        fangle += 2 * M_PI;
//                        cout << fangle << endl;
//                        cout << rotate_angle << endl;
                    }
                    CHECK_LT(fabs(fangle - rotate_angle), 1e-3);
                    CHECK_LT(fabs(cur_bb.BoxCenter()[0] - curx), 1e-3);
                    CHECK_LT(fabs(cur_bb.BoxCenter()[1] - cury), 1e-3);
                    BoxPosToBoxIndex(x_st_ed, y_st_ed, delta_x, delta_y, delta_rotation, cur_bb.BoxCenter()[0], cur_bb.BoxCenter()[1], fangle, &cur_idx_test);
                    CHECK_EQ(cur_idx_test, box_index);
                }

//                if (save_path)
//                {
//                    const std::string& output_dir_prefix = *save_path;
//                    char number[10];
//                    sprintf(number, "_%05d", box_index);
//                    string obb_name = output_dir_prefix + "_" + number + "_feat_debug_all_obb.ply";
//                    cpu_tsdf::SaveOrientedBoundingbox(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);
//                }

                cpu_tsdf::TSDFFeature cur_feat;
                if (cpu_tsdf::GenerateOneSample(tsdf_model, cur_bb, template_occpy_ratio, mesh_min_weight, &cur_feat))
                {
                     ////////
                    vector<vector<float>> feat_vec(1);
                    cur_feat.GetFeatureVector(&template_feature,  &(feat_vec[0]), mesh_min_weight, true);
                    vector<int> input_labels(1, -1);
                    vector<int> output_label;
                    vector<vector<float>> output_score;
                    float accuracy;
                    //svm.SVMPredict(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
                    svm.SVMPredict_Primal(feat_vec, input_labels, "", &output_label, &output_score, &accuracy, save_path);
                    int cur_label = output_label[0];
                    float cur_score = output_score[0][0];
//                    fprintf(hf, "pos: %f %f, angle: %f\n, feat_size: %d\n", curx, cury, rotate_angle, feat_vec[0].size());
//                    for (int tti = 0; tti < feat_vec[0].size(); ++tti)
//                    {
//                        fprintf(hf, "%d:%f,  ", tti, feat_vec[0][tti]);
//                    }
//                    fprintf(hf, "\n");
                    fprintf(hf, "%d %d %f\n", box_index, cur_label, cur_score);
                    fflush(hf);

//                    if (box_index == 3178067)
//                    {
//                        using namespace std;
//                        cout << box_index << endl;
//                        cout << cur_label << endl;
//                        cout << cur_score << endl;
////                        char ch;
////                        cin >> ch;
//                    }

                    ////////////////////////////

                    if ( cur_score >= 0)
                    {
                        //output the hard negative samples
                        if (save_path)
                        {
                            /********/
                            cout << "save " << box_index << "th bb" << endl;
                            char weight_num[10];
                            sprintf(weight_num, "%.02f", mesh_min_weight);
                            if (save_path)
                            {
                                using namespace std;
                                char cscore[15];
                                char clabel[5];
                                sprintf(cscore, "%.04f", cur_score);
                                sprintf(clabel, "%02d", cur_label);
                                const std::string output_dir_prefix = *save_path + "_detected_label_" + clabel + "_score_" + cscore;
                                char number[10];
                                sprintf(number, "_%05d", box_index);
                                string feat_name = output_dir_prefix + "_boxidx_" + number + "_feat_" + weight_num + ".txt";
                                vector<float> neg_out_feat;
//                                cur_feat.GetFeatureVector(&template_feature,  &neg_out_feat, mesh_min_weight, true);
//                                utility::OutputVector(feat_name, neg_out_feat);

//                                string feat_name = output_dir_prefix + "_boxidx_" + number + "_obb.txt";
//                               cpu_tsdf::WriteOrientedBoundingboxPly()
                                //assert(neg_out_feat.size() == out_feat_template.size());

//                                cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                                SliceTSDFWithBoundingbox_NearestNeighbor(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                                string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_NN.bin";
//                                std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                boost::archive::binary_oarchive oa(os);
//                                oa << *(cur_feat_tsdf);
//                                string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + "_NN.ply";
//                                cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);

//                                {
//                                    cpu_tsdf::TSDFHashing::Ptr cur_feat_tsdf(new TSDFHashing);
//                                    SliceTSDFWithBoundingbox(&tsdf_model, cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, cur_feat.VoxelLengths()[0], cur_feat_tsdf.get());
//                                    string tsdf_bin_name = output_dir_prefix + "_" + number + "_tsdf_bin_Linear.bin";
//                                    std::ofstream os(tsdf_bin_name, std::ios_base::out);
//                                    boost::archive::binary_oarchive oa(os);
//                                    oa << *(cur_feat_tsdf);
//                                    string tsdf_mesh_name = output_dir_prefix + "_" + number + "_tsdf_mesh_" + weight_num + "_linear.ply";
//                                    cpu_tsdf::WriteTSDFMesh(cur_feat_tsdf, mesh_min_weight, tsdf_mesh_name, false);
//                                }

                                string obb_name = output_dir_prefix + "_" + number + "_feat_obb.ply";
                                cpu_tsdf::WriteOrientedBoundingboxPly(cur_bb.bb_orientation, cur_bb.bb_offset, cur_bb.bb_sidelengths, obb_name);




                            }  //
                            cout << "save " << box_index << "th bb finished" << endl;
                            /*******/
                        }
                    }
                    ////////////////////////////
                    //std::cout << "hard neg box idx: " << box_index << std::endl;
                    (*res)[box_index/delta_box_index] = (ClassificationResult(box_index, cur_label, cur_score));
                }
            }
    if (save_path)
    {
        fclose(hf);
    }
    }
    catch(std::exception & e)
    {
        cout << "exception in thread" << endl;
        cout << e.what() << endl;
    }
    catch (...)
    {
        cout << "unknown exception in thread" << endl;
    }

    return;

}


void cpu_tsdf::SlidingBoxDetection_Parrellel(const cpu_tsdf::TSDFHashing &tsdf_model,
                                             const cpu_tsdf::OrientedBoundingBox &template_bb, const SVMWrapper &svm,
                                             const float template_occpy_ratio,
                                             const int total_thread,
                                             const float delta_x, const float delta_y, const float delta_rotation,
                                             float *x_st_ed, float *y_st_ed, std::vector<cpu_tsdf::ClassificationResult> *res, float mesh_min_weight, const std::string *save_path)
{
    using namespace boost::threadpool;
    Eigen::Vector3f min_pt, max_pt;
    // tsdf_model.RecomputeBoundingBoxInVoxelCoord();
    // tsdf_model.Re
    tsdf_model.getBoundingBoxInWorldCoord(min_pt, max_pt);
    cout << "min_pt: " << min_pt << endl;
    cout << "max_pt: " << max_pt << endl;
    cout << "parallel x sample num: " << floor((max_pt[0] - min_pt[0])/delta_x) + 1  << endl;
    cout << "parallel y sample num: " << floor((max_pt[1] - min_pt[1])/delta_y) + 1  << endl;
    cout << "parallel rotate sample num: " << floor((2*M_PI - 1e-5)/delta_rotation) + 1 << endl;
//    char ch;
//    cin >> ch;
    int x_s_num = floor((max_pt[0] - min_pt[0])/delta_x) + 1;
    int y_s_num = floor((max_pt[1] - min_pt[1])/delta_y) + 1;
    int rotate_s_num = floor((2*M_PI - 1e-5)/delta_rotation) + 1;
    x_st_ed[0] = min_pt[0];
    x_st_ed[1] = max_pt[0];
    y_st_ed[0] = min_pt[1];
    y_st_ed[1] = max_pt[1];
    std::vector<std::vector<cpu_tsdf::ClassificationResult>> all_res(rotate_s_num);
    std::vector<std::string> save_paths(rotate_s_num);

    {
        /*
    void cpu_tsdf::SlidingBoxDetection_OneRotationAngle(const cpu_tsdf::TSDFHashing &tsdf_model,
                                                        const cpu_tsdf::OrientedBoundingBox &template_bb, SVMWrapper &svm,
                                                        const float delta_x, const float delta_y, const float delta_rotation, const float rotate_angle,
                                                        float *x_st_ed, float *y_st_ed,
                                                        std::map<int, cpu_tsdf::ClassificationResult> * res, float mesh_min_weight, const std::string *save_path)
         */
        // Create a thread pool.
        // pool tp(total_thread);
        pool tp(1);
        // Add some tasks to the pool.
        int cnt = -1;
        //for (float rotate_angle = 0; rotate_angle <= 0; rotate_angle += delta_rotation)
        //for (float rotate_angle = 0; rotate_angle < 2 * M_PI - 1e-5; rotate_angle += delta_rotation)
        for (float rotate_angle = delta_rotation* 73; rotate_angle < delta_rotation*73 + 1e5; rotate_angle += delta_rotation)
        {
            cnt++;
            cnt = 73;
            char temp[128] = {0};
            sprintf(temp, "_%02dth_rotate_angle_%.5f", cnt, rotate_angle);
            if (save_path)
            {
                save_paths[cnt] = *save_path + temp;
            }
            else
            {
                save_paths[cnt] = "";
            }
//            tp.schedule(std::bind(&SlidingBoxDetection_OneRotationAngle, std::cref(tsdf_model),
//                                  std::cref(template_bb), std::cref(svm),
//                                  delta_x, delta_y, delta_rotation, rotate_angle,
//                                  std::cref(min_pt), std::cref(max_pt),
//                                  &(all_res[cnt]), mesh_min_weight, (!save_paths[cnt].empty())? &(save_paths[cnt]):NULL));
            tp.schedule(std::bind(&SlidingBoxDetection_OneRotationAngle2, std::cref(tsdf_model),
                                  std::cref(template_bb),
                                  template_occpy_ratio,
                                  std::cref(svm),
                                  delta_x, delta_y, delta_rotation, rotate_angle,
                                  std::cref(min_pt), std::cref(max_pt),
                                  &(all_res[cnt]), mesh_min_weight, (!save_paths[cnt].empty())? &(save_paths[cnt]):NULL));
        }
        // Leave this function and wait until all tasks are finished.
    }
    // combine the results into one vector
    cout << "combining results" << endl;
    res->resize(x_s_num * y_s_num * rotate_s_num);
    int res_index = -1;
    for (int i = 0; i <x_s_num * y_s_num; ++i)
        for (int rot_i = 0; rot_i < rotate_s_num; ++rot_i)
        {
            res_index++;
            (*res)[res_index] = all_res[rot_i][i];
        }
    cout << "writing to file" << endl;
    // write out to file
    FILE* hf = NULL;
    if (save_path)
    {
        hf = fopen(((*save_path) + "_SlidingBoxDetectionResults_Parallel_Final.txt").c_str(), "w");
        fprintf(hf, "index label score\n");
    }
    fprintf(hf, "min_pt: %f %f %f\n", min_pt[0], min_pt[1], min_pt[2]);
    fprintf(hf, "max_pt: %f %f %f\n", max_pt[0], max_pt[1], max_pt[2]);
    fprintf(hf, "delta_x, delta_y, delta_r: %f %f %f\n", delta_x, delta_y, delta_rotation);
    fprintf(hf, "scene_vx, scene_vy, scene_vr: %d %d %d\n", x_s_num, y_s_num, rotate_s_num);
    for (int i = 0; i < (*res).size(); ++i)
    {
        if ((*res)[i].index >= 0)
        {
            fprintf(hf, "%d %d %f\n", (*res)[i].index, (*res)[i].label, (*res)[i].score);
        }
    }
    fclose(hf);
}
