/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "oriented_boundingbox.h"
#include <iostream>
#include <Eigen/Eigen>
#include <pcl/io/ply_io.h>
#include "utility.h"
#include "common/utilities/eigen_utility.h"

namespace tsdf_utility {
void OrientedBoundingBox::ComputeVertices(Eigen::Matrix<float, 3, 8> *obb_vertices) const {
    static const float unit_cube[24] = {-0.5, -0.5, 0,
                                          -0.5, -0.5, 1,
                                          -0.5, 0.5, 0,
                                          -0.5, 0.5, 1,
                                          0.5, -0.5, 0,
                                          0.5, -0.5, 1,
                                          0.5, 0.5, 0,
                                          0.5, 0.5, 1};
    const Eigen::Map<const Eigen::Matrix<float, 3, 8>> unit_cube_mat(unit_cube);
    *obb_vertices = transform_ * unit_cube_mat;
}

OrientedBoundingBox::OrientedBoundingBox() {
    transform_.setIdentity();
}

OrientedBoundingBox::OrientedBoundingBox(const Eigen::Vector3f &side_lengths, const Eigen::Vector3f &bottom_center, const double angle) {
    Eigen::Matrix3f rotation;
    rotation << cos(angle), -sin(angle), 0,
            sin(angle), cos(angle), 0,
            0,0,1;
    transform_.linear() = rotation * side_lengths.asDiagonal();
    transform_.translation() = bottom_center;
}

OrientedBoundingBox::OrientedBoundingBox(const Eigen::Vector3f &side_lengths, float cx, float cy, float cbz, const double angle)
    :OrientedBoundingBox(side_lengths, Eigen::Vector3f(cx, cy, cbz), angle) {}

const Eigen::Affine2f OrientedBoundingBox::Affine2D() const
{
    Eigen::Matrix3f trans2d = Eigen::Matrix3f::Identity();
    trans2d.block(0, 0, 2, 2) = transform_.matrix().block(0, 0, 2, 2);
    trans2d.block(0, 2, 2, 1) = transform_.matrix().block(0, 3, 2, 1);
    Eigen::Affine2f res;
    res.matrix() = trans2d;
    return res;
}

const Eigen::Affine3f OrientedBoundingBox::AffineTransform() const
{
    Eigen::Affine3f res;
    res.matrix() = Eigen::Matrix4f::Identity();
    res.matrix().block(0, 0, 3, 4) = transform_.matrix();
    return res;
}

const Eigen::Affine3f OrientedBoundingBox::AffineTransformOriginAsCenter() const
{
    Eigen::Affine3f res;
    res.matrix() = Eigen::Matrix4f::Identity();
    res.matrix().block(0, 0, 3, 4) = transform_.matrix();
    res.matrix().block(0, 3, 3, 1) = transform_.matrix().block(0, 3, 3, 1) + 0.5 * transform_.matrix().block(0, 2, 3, 1);
    return res;
}

int OrientedBoundingBox::WriteToPly(const std::string &filename) const {
    pcl::PolygonMesh mesh;
    ToPolygonMesh(&mesh);
    return pcl::io::savePLYFileBinary(filename, mesh);
}

int OrientedBoundingBox::WriteToFile(const std::string &filename) const
{
    std::ofstream ofs(filename);
    ofs << *this;
    return bool(ofs);
}

OrientedBoundingBox OrientedBoundingBox::ExtendSides(const Eigen::Vector3f &extension) const
{
    // extends the sides of a bounding box
    // x and y will be extended along both directions by extension[0] and extension[1], z side will be extended only along the positive direction by extension[2]
    // (i.e. the bounding box will not be extended below the ground plane)
    Eigen::Vector3f old_side_lengths = transform_.linear().colwise().norm();
    Eigen::Vector3f new_side_lengths = old_side_lengths;
    new_side_lengths[0] += extension[0] * 2;
    new_side_lengths[1] += extension[1] * 2;
    new_side_lengths[2] += extension[2];
    OrientedBoundingBox res = *this;
    res.transform_.linear() *= (new_side_lengths.cwiseQuotient(old_side_lengths)).asDiagonal();
    return res;
}

OrientedBoundingBox OrientedBoundingBox::ExtendSidesByPercent(const Eigen::Vector3f &extension_percent) const
{
    Eigen::Vector3f side_percent(1, 1, 1);
    side_percent[0] += extension_percent[0] * 2;
    side_percent[1] += extension_percent[1] * 2;
    side_percent[2] += extension_percent[2];
    OrientedBoundingBox res = *this;
    res.transform_.linear() *= side_percent.asDiagonal();
    return res;
}

double OrientedBoundingBox::AngleRangeTwoPI() const {
    const Eigen::Matrix3f orient = Orientations();
    double ang = atan2(double(orient(1, 0)), double(orient(0, 0)));
    return cpu_tsdf::WrapAngleRange2PI(ang);
}

double OrientedBoundingBox::AngleRangePosNegPI() const
{
    const Eigen::Matrix3f orient = Orientations();
    double ang = atan2(double(orient(1, 0)), double(orient(0, 0)));
    return ang;
}

void OrientedBoundingBox::ToPolygonMesh(pcl::PolygonMesh *pmesh) const {
    pcl::PolygonMesh& mesh = *pmesh;
    Eigen::Matrix<float, 3, 8> obb_vertices;
    ComputeVertices(&obb_vertices);

    pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
    for (size_t i = 0; i < 8; i++)
    {
        pcl::PointXYZRGB pt_final;
        pt_final.x = obb_vertices(0, i);
        pt_final.y = obb_vertices(1, i);
        pt_final.z = obb_vertices(2, i);
        pt_final.r = pt_final.g = pt_final.b = 255;
        vertices_new.push_back(pt_final);
    }
    // blue: z, green: y, red: x
    vertices_new[0].r = 0; vertices_new[0].g = 0; vertices_new[0].b = 255;
    vertices_new[1].r = 0; vertices_new[1].g = 0; vertices_new[1].b = 255;
    vertices_new[2].r = 0; vertices_new[2].g = 255; vertices_new[2].b = 0;
    vertices_new[3].r = 0; vertices_new[3].g = 255; vertices_new[3].b = 0;
    vertices_new[4].r = 255; vertices_new[4].g = 0; vertices_new[4].b = 0;
    vertices_new[5].r = 255; vertices_new[5].g = 0; vertices_new[5].b = 0;
    pcl::toPCLPointCloud2(vertices_new, mesh.cloud);

    mesh.polygons.resize(12);
    for (size_t i = 0; i < 12; i++)
    {
        mesh.polygons[i].vertices.resize(3);
    }
    mesh.polygons[0].vertices[0] = 0; mesh.polygons[0].vertices[1] = 1; mesh.polygons[0].vertices[2] = 3;
    mesh.polygons[1].vertices[0] = 0; mesh.polygons[1].vertices[1] = 3; mesh.polygons[1].vertices[2] = 2;
    mesh.polygons[2].vertices[0] = 6; mesh.polygons[2].vertices[1] = 7; mesh.polygons[2].vertices[2] = 5;
    mesh.polygons[3].vertices[0] = 6; mesh.polygons[3].vertices[1] = 5; mesh.polygons[3].vertices[2] = 4;
    mesh.polygons[4].vertices[0] = 0; mesh.polygons[4].vertices[1] = 4; mesh.polygons[4].vertices[2] = 5;
    mesh.polygons[5].vertices[0] = 0; mesh.polygons[5].vertices[1] = 5; mesh.polygons[5].vertices[2] = 1;
    mesh.polygons[6].vertices[0] = 1; mesh.polygons[6].vertices[1] = 5; mesh.polygons[6].vertices[2] = 7;
    mesh.polygons[7].vertices[0] = 1; mesh.polygons[7].vertices[1] = 7; mesh.polygons[7].vertices[2] = 3;
    mesh.polygons[8].vertices[0] = 3; mesh.polygons[8].vertices[1] = 7; mesh.polygons[8].vertices[2] = 6;
    mesh.polygons[9].vertices[0] = 3; mesh.polygons[9].vertices[1] = 6; mesh.polygons[9].vertices[2] = 2;
    mesh.polygons[10].vertices[0] = 2; mesh.polygons[10].vertices[1] = 6; mesh.polygons[10].vertices[2] = 4;
    mesh.polygons[11].vertices[0] = 2; mesh.polygons[11].vertices[1] = 4; mesh.polygons[11].vertices[2] = 0;
}

OrientedBoundingBox NewOBBFromOld(const cpu_tsdf::OrientedBoundingBox &old_obb)
{
    Eigen::AffineCompact3f trans;
    trans.linear() = old_obb.bb_orientation * old_obb.bb_sidelengths.asDiagonal();
    trans.translation() = old_obb.BoxBottomCenter();
    Eigen::Vector3i sample_size = (old_obb.bb_sidelengths.cwiseQuotient(old_obb.voxel_lengths)).cast<int>() + Eigen::Vector3i(1, 1, 1);
    return OrientedBoundingBox(trans);
}

std::ostream &operator <<(std::ostream &ofs, const OrientedBoundingBox &obb)
{
    // ofs << obb.transform_.matrix() << std::endl;
    utility::operator <<(ofs, obb.transform_.matrix());
    return ofs;
}

std::istream &operator >>(std::istream &ifs, OrientedBoundingBox &obb)
{
    Eigen::MatrixXf mat;
    utility::operator >>(ifs, mat);
    obb.transform_.matrix() = mat;
    return ifs;
}

std::vector<OrientedBoundingBox> NewOBBsFromOlds(const std::vector<cpu_tsdf::OrientedBoundingBox> &old_obbs)
{
    std::vector<OrientedBoundingBox> res;
    for (int i = 0; i < old_obbs.size(); ++i) {
        res.push_back(NewOBBFromOld(old_obbs[i]));
    }
    return res;
}

void OutputAnnotatedOBB(const std::vector<std::vector<OrientedBoundingBox> > &obbs, const std::string &filename)
{
    std::ofstream ofs(filename);
    for (int i = 0; i < obbs.size(); ++i) {
        for (int j = 0; j < obbs[i].size(); ++j) {
            ofs << i << std::endl;
            ofs << obbs[i][j] << std::endl;
        }
    }
}

void OutputAnnotatedOBB(const std::vector<OrientedBoundingBox>& obb_vec, const std::vector<int>& sample_model_idx, const std::string& filename) {
    std::vector<std::vector<OrientedBoundingBox>> obbs;
    std::map<int, int> category_idx_map;
    int category = 0;
    for (int i = 0; i < sample_model_idx.size(); ++i) {
        const tsdf_utility::OrientedBoundingBox& cur_obb = obb_vec[i];
        auto res = category_idx_map.find(category);
        int idx = -1;
        if (res == category_idx_map.end()) {
            category_idx_map[category] = obbs.size();
            idx = obbs.size();
            (obbs).push_back(std::vector<tsdf_utility::OrientedBoundingBox>());
        } else {
            idx = res->second;
        }
        obbs[idx].push_back(cur_obb);
    }
    OutputAnnotatedOBB(obbs, filename);
}

void InputAnnotatedOBB(const std::string &filename, std::vector<std::vector<OrientedBoundingBox> > *obbs) {
    std::ifstream ifs(filename);
    int category = 0;
    while(ifs >> category) {
        tsdf_utility::OrientedBoundingBox cur_obb;
        ifs >> cur_obb;
        obbs->resize(category + 1);
        (*obbs)[category].push_back(cur_obb);
    }
}

void OutputOBBsAsPly(const std::vector<OrientedBoundingBox> &obbs, const std::string &filename)
{
    for (int i = 0; i < obbs.size(); ++i) {
        std::string  cur_file = filename + "_" + utility::int2str(i, 3) + ".ply";
        obbs[i].WriteToPly(cur_file);
    }
}

cpu_tsdf::OrientedBoundingBox OldOBBFromNew(const OrientedBoundingBox &obb)
{
    cpu_tsdf::OrientedBoundingBox oldobb;
    cpu_tsdf::AffineToOrientedBB(obb.AffineTransformOriginAsCenter(), &oldobb);
    return oldobb;
}

}
