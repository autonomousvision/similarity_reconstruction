/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_io.h"

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
//#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Eigen>
#include <matio.h>

#include "tsdf_operation/tsdf_align.h"
#include "tsdf_operation/tsdf_transform.h"
#include "tsdf_operation/tsdf_slice.h"
#include "common/utilities/pcl_utility.h"
#include "common/utilities/eigen_utility.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "common/utilities/eigen_utility.h"


using namespace std;
namespace bfs = boost::filesystem;
using namespace bfs;



namespace cpu_tsdf {

bool WriteOrientedBoundingboxPly(const OrientedBoundingBox &obb, const std::string &filename)
{
    return WriteOrientedBoundingboxPly(obb.bb_orientation, obb.bb_offset, obb.bb_sidelengths, filename);
}

bool WriteOrientedBoundingboxPly(const Eigen::Matrix3f &orientation, const Eigen::Vector3f &offset, const Eigen::Vector3f &lengths, const std::string &filename)
{
    pcl::PolygonMesh mesh;
    Eigen::Vector3f box_vertices[8];
    ComputeOrientedBoundingboxVertices(orientation, offset, lengths, box_vertices);
    pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
    for (size_t i = 0; i < 8; i++)
    {
        pcl::PointXYZRGB pt_final;
        pt_final.x = box_vertices[i][0];
        pt_final.y = box_vertices[i][1];
        pt_final.z = box_vertices[i][2];
        pt_final.r = 255;
        pt_final.g = 255;
        pt_final.b = 255;
        vertices_new.push_back(pt_final);
    }
    // blue: z, green: y, red: x
    vertices_new[0].r = 0; vertices_new[0].g = 0; vertices_new[0].b = 255;
    vertices_new[1].r = 0; vertices_new[1].g = 0; vertices_new[1].b = 255;
    vertices_new[2].r = 0; vertices_new[2].g = 255; vertices_new[2].b = 0;
    vertices_new[3].r = 0; vertices_new[3].g = 255; vertices_new[3].b = 0;
    vertices_new[4].r = 255; vertices_new[4].g = 0; vertices_new[4].b = 0;
    vertices_new[5].r = 255; vertices_new[5].g = 0; vertices_new[5].b = 0;
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
    pcl::toPCLPointCloud2(vertices_new, mesh.cloud);
    pcl::io::savePLYFile(filename, mesh);
    return true;
}


bool WriteOrientedBoundingboxesPly(const std::vector<OrientedBoundingBox> &obb, const string &filename, const std::vector<double>& outlier_gammas)
{
    for (int i = 0; i < obb.size(); ++i)
    {
        bfs::path cur_path = bfs::path(filename);
        char tmp_char[50];
        if (outlier_gammas.empty() || outlier_gammas.at(i) <= 1e-5)
        {
            sprintf(tmp_char, "_obb_%010d.ply",i);
        }
        else
        {
            sprintf(tmp_char, "_obb_%010d_outlier_gamma_%10.6f.ply", i, outlier_gammas.at(i));
        }
        WriteOrientedBoundingboxPly(obb[i], cur_path.replace_extension(tmp_char).string());
    }
}

bool WriteOrientedBoundingBoxes(const string &filename, const std::vector<OrientedBoundingBox> obbs, const std::vector<int> sample_model_assign, const std::vector<bool> is_train_sample)
{
    using namespace std;
    ofstream ofs(filename);
    const int sample_num = obbs.size();
    ofs << sample_num << endl;
    for (int i  = 0; i < sample_num; ++i)
    {
        const OrientedBoundingBox& cur_box = obbs[i];
        ofs << i << endl;
        ofs << "offset: " << cur_box.bb_offset[0] << " "
            << cur_box.bb_offset[1] << " "
            << cur_box.bb_offset[2] << endl;
        ofs << "orientation: " << endl;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                ofs << cur_box.bb_orientation.coeff(r, c) << " ";
            }
            ofs << endl;
        }
        ofs << "side_lengths: " << cur_box.bb_sidelengths[0] << " "
            << cur_box.bb_sidelengths[1] << " "
            << cur_box.bb_sidelengths[2] << endl;
        ofs << "voxel_length: " << cur_box.voxel_lengths[0] << " "
            << cur_box.voxel_lengths[1]  << " "
            << cur_box.voxel_lengths[2] << endl;
        ofs << "model: " << sample_model_assign[i] << endl;
        if (!is_train_sample.empty())
        {
            ofs << "train: " << is_train_sample[i] << endl;
        }
        else
        {
            ofs << "train: " << true << endl;
        }
        ofs << endl;
    }
    return true;
}


bool ReadOrientedBoundingBoxes(const string &filename, std::vector<OrientedBoundingBox> *obbs, std::vector<int> *sample_model_assign, std::vector<bool> *is_train_sample)
{
    using namespace std;
    if (!bfs::exists(bfs::path(filename))) return false;
    ifstream ifs(filename);
    int sample_num;
    ifs >> sample_num;
    ifs.get();
    for (int i = 0; i < sample_num; ++i)
    {
        string dummy;
        int cur_i;
        ifs >> cur_i;
        if (!ifs) break;
        ifs.get();
        float offset_x, offset_y, offset_z;
        ifs>>dummy>>offset_x>>offset_y>>offset_z;
        Eigen::Vector3f offset(offset_x, offset_y, offset_z);
        ifs >> dummy; ifs.get();

        Eigen::Matrix3f orientation;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                float cur_number;
                ifs >> cur_number;
                orientation.coeffRef(r, c) = cur_number;
            }
            ifs.get();
        }

        Eigen::Vector3f side_lengths;
        ifs >> dummy>>side_lengths[0] >> side_lengths[1] >> side_lengths[2]; ifs.get();

        Eigen::Vector3f voxel_lengths;
        ifs >> dummy >> voxel_lengths[0] >> voxel_lengths[1] >> voxel_lengths[2]; ifs.get();
        int model;
        ifs >> dummy >> model; ifs.get();
        bool is_train = true;
        if (ifs.peek() == 't')
        {
            ifs >> dummy >> is_train; ifs.get();
        }
        ifs.get();

        OrientedBoundingBox cur_box;
        cur_box.bb_offset = offset;
        cur_box.bb_orientation = orientation;
        cur_box.bb_sidelengths = side_lengths;
        cur_box.voxel_lengths = voxel_lengths;
        // cur_box.is_trainining_sample = is_train;
        obbs->push_back(cur_box);
        sample_model_assign->push_back(model);
        if (is_train_sample)
            is_train_sample->push_back(is_train);

    }
    return true;
}

bool WriteTSDFModel(
        TSDFHashing::ConstPtr tsdf_model,
        const std::string &output_filename,
        bool save_tsdf_bin,
        bool save_mesh,
        float mesh_min_weight)
{
    string output_dir = bfs::path(output_filename).remove_filename().string();
    string output_tsdf_gridfilename =
                    (bfs::path(output_dir) / (bfs::path(output_filename).stem().string()
                                              + "_tsdf_grid.ply"
                                              )).string();
    // tsdf_model->OutputTSDFGrid(output_tsdf_gridfilename, NULL, NULL);
    if (save_tsdf_bin)
    {
        std::cout << "saving TSDF binary file" << std::endl;
        string output_tsdffilename =
                (bfs::path(output_dir) / (bfs::path(output_filename).stem().string()
                                          + "_tsdf.bin"
                                          )).string();
        std::cout << "save tsdf file path: " << output_tsdffilename << std::endl;
        std::ofstream os(output_tsdffilename, std::ios_base::out);
        boost::archive::binary_oarchive oa(os);
        oa << *(tsdf_model);
        std::cout << "saving TSDF binary file finished" << std::endl;

    }
    if (save_mesh)
    {
        string output_plyfilename =
                (bfs::path(output_dir) / (bfs::path(output_filename).stem().string()
                                          + "_mesh.ply")).string();
        WriteTSDFMesh(tsdf_model, mesh_min_weight, output_plyfilename, false);
    }
    return true;
}

bool WriteTSDFModels(const std::vector<TSDFHashing::Ptr> &tsdf_models,
                              const string &output_filename,
                              bool save_tsdf_bin,
                              bool save_mesh,
                              float mesh_min_weight,
                              const std::vector<double>& outlier_gammas)
{
    string output_dir = bfs::path(output_filename).remove_filename().string();
    for (int ti = 0; ti < tsdf_models.size(); ++ti)
    {
        string current_output_filename =
                (bfs::path(output_dir) / (bfs::path(output_filename).stem().string()
                                          + "_tsdfnum_" + utility::int2str(ti, 5)
                                          + ".ply")).string();
        if (!outlier_gammas.empty() && outlier_gammas.at(ti) > 1e-5)
        {
            current_output_filename = bfs::path(current_output_filename).replace_extension(std::string("outlier_") + utility::double2str(outlier_gammas[ti]) + ".ply").string();
        }
        WriteTSDFModel(tsdf_models[ti], current_output_filename,
                      save_tsdf_bin, save_mesh, mesh_min_weight);
    }
    return true;
}

void WriteTSDFMesh(
        TSDFHashing::ConstPtr tsdf_model,
        float mesh_min_weight,
        const string &output_filename,
        const bool save_ascii)
{
    std::cout << "begin marching cubes" << std::endl;
    pcl::PolygonMesh::Ptr mesh = TSDFToPolygonMesh(tsdf_model, mesh_min_weight, 0.00005);
    std::cout << "save model at ply file path: " << output_filename << std::endl;
    if (save_ascii)
        pcl::io::savePLYFile (output_filename, *mesh);
    else
        pcl::io::savePLYFileBinary (output_filename, *mesh);
    std::cout << "save finished" << std::endl;
}

bool WriteAffineTransformsAndTSDFs(
        const TSDFHashing &scene_tsdf,
        const std::vector<Eigen::Affine3f> &affine_transforms,
        const TSDFGridInfo& tsdf_info,
        const string &save_path,
        bool save_text_data)
{
    const int sample_num = affine_transforms.size();
    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
    ExtractSamplesFromAffineTransform(
            scene_tsdf,
            affine_transforms,
            tsdf_info,
            &samples,
            &weights);

    for (int i = 0; i < sample_num; ++i)
    {
        const bfs::path prefix(save_path);
        std::string cur_save_path = (prefix.parent_path()/prefix.stem()).string() + "_obb_" + boost::lexical_cast<string>(i) + ".ply";
        // 1. save obb
        Eigen::Matrix3f test_r;
        Eigen::Vector3f test_scale;
        Eigen::Vector3f test_trans;
        utility::EigenAffine3fDecomposition(
                    affine_transforms[i],
                    &test_r,
                    &test_scale,
                    &test_trans);
        WriteOrientedBoundingboxPly(test_r, test_trans - (test_r * test_scale.asDiagonal() * Eigen::Vector3f::Ones(3, 1))/2.0f, test_scale, cur_save_path);
        // 2. save TSDF
        TSDFHashing::Ptr cur_tsdf(new TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
        samples.col(i),
        weights.col(i),
        tsdf_info,
        cur_tsdf.get());
        string save_path = (prefix.parent_path()/prefix.stem()).string() + "_affinetrans_" + boost::lexical_cast<string>(i) + ".ply";
        TSDFHashing::Ptr transformed_tsdf(new TSDFHashing);
        float voxel_len = (scene_tsdf.voxel_length());
        TransformTSDF(*cur_tsdf, affine_transforms[i], transformed_tsdf.get(), &voxel_len);
        WriteTSDFModel(transformed_tsdf, save_path, false, true, tsdf_info.min_model_weight());

        string save_path_canonical = (prefix.parent_path()/prefix.stem()).string() + "_canonical_" + boost::lexical_cast<string>(i) + ".ply";
        WriteTSDFModel(cur_tsdf, save_path_canonical, false, true, tsdf_info.min_model_weight());

        // 3. save data vector, weight vector, data world coordinate (all in canonical positions)
        if (save_text_data)
        {
            string text_save_path = (prefix.parent_path()/prefix.stem()).string() + "_textdata_canonical_" + boost::lexical_cast<string>(i) + ".txt";
            Eigen::MatrixXf save_mat(samples.rows(), 8);
            save_mat.setZero();
            save_mat.col(0) = samples.col(i); // data
            save_mat.col(1) = weights.col(i); // weight

            TSDFHashing tmp_tsdf;
            std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f>> idx_worldpos;
            ConvertDataVectorToTSDFWithWeightAndWorldPos(samples.col(i),
                                                         weights.col(i),
                                                         tsdf_info,
                                                         &tmp_tsdf,
                                                         &idx_worldpos);
            for (std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f>>::const_iterator citr = idx_worldpos.begin(); citr !=  idx_worldpos.end(); ++citr)
            {
                CHECK_LT(citr->first, save_mat.rows());
                save_mat.coeffRef(citr->first, 2 + 0) = (citr->second).first[0];
                save_mat.coeffRef(citr->first, 2 + 1) = (citr->second).first[1];
                save_mat.coeffRef(citr->first, 2 + 2) = (citr->second).first[2];

                save_mat.coeffRef(citr->first, 2 + 3) = (citr->second).second[0];
                save_mat.coeffRef(citr->first, 2 + 4) = (citr->second).second[1];
                save_mat.coeffRef(citr->first, 2 + 5) = (citr->second).second[2];
            }
            utility::WriteEigenMatrix(save_mat, text_save_path);
        }
    }
    return true;
}

void WriteTSDFsFromMatNoWeight(const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, const Eigen::Vector3i &boundingbox_size, const float voxel_length, const Eigen::Vector3f &offset, const float max_dist_pos, const float max_dist_neg, const string &save_filepath)
{
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    cpu_tsdf::ConvertDataMatrixToTSDFsNoWeight(voxel_length,
                                               offset,
                                               max_dist_pos,
                                               max_dist_neg,
                                               boundingbox_size,
                                               data_mat,
                                               &projected_tsdf_models
                                               );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_modelply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "_frommat_noweight.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
}

void WriteTSDFsFromMatWithWeight(const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat, const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat, const Eigen::Vector3i &boundingbox_size, const float voxel_length, const Eigen::Vector3f &offset, const float max_dist_pos, const float max_dist_neg, const string &save_filepath)
{
    std::vector<cpu_tsdf::TSDFHashing::Ptr> projected_tsdf_models;
    cpu_tsdf::ConvertDataMatrixToTSDFs(voxel_length,
                                               offset,
                                               max_dist_pos,
                                               max_dist_neg,
                                               boundingbox_size,
                                               data_mat,
                                               weight_mat,
                                               &projected_tsdf_models
                                               );
    string output_dir = bfs::path(save_filepath).remove_filename().string();
    string output_modelply =
            (bfs::path(output_dir) / (bfs::path(save_filepath).stem().string()
                                      + "_frommat_withweight.ply")).string();
    cpu_tsdf::WriteTSDFModels(projected_tsdf_models, output_modelply, false, true, 0);
}

void WriteTSDFsFromMatWithWeight_Matlab(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &data_mat,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &weight_mat,
        const TSDFGridInfo &tsdf_info, const string &save_filepath, const std::string& var_suffix)
{
    //debug
    utility::WriteEigenMatrix(data_mat, save_filepath + "_debug_data_mat.txt");
    utility::WriteEigenMatrix(weight_mat, save_filepath + "_debug_weight_mat.txt");
    WriteTSDFsFromMatWithWeight(data_mat, weight_mat, tsdf_info, save_filepath + "_debug.ply");
    for (int i = 0; i < data_mat.cols(); ++i)
    {
        WriteTSDFFromVectorWithWeight_Matlab(data_mat.col(i), weight_mat.col(i), tsdf_info,
                                            save_filepath, var_suffix + string("_") + utility::int2str(i));
    }
}

bool WriteTSDFFromVectorWithWeight_Matlab(const Eigen::SparseVector<float> &data_mat, const Eigen::SparseVector<float> &weight_mat,
                                          const TSDFGridInfo &tsdf_info, const string &save_filepath, const string& var_suffix)
{
    /* create some matrices*/
    const Eigen::Vector3i& bbsize = tsdf_info.boundingbox_size();
    const int tx = bbsize[0];
    const int ty = bbsize[1];
    const int tz = bbsize[2];
    double datamat_3d[bbsize[0]][bbsize[1]][bbsize[2]];
    double weightmat_3d[bbsize[0]][bbsize[1]][bbsize[2]];
    int cnt = 0;
    for(int i=0;i<tx;i++)
        for(int j=0;j<ty;j++)
            for (int k=0;k<tz;k++)
            {
                // to ensure matlab reads as x, y, z order
            //    datamat_3d[k][j][i] = data_mat.coeff(cnt);
            //    weightmat_3d[k][j][i] = weight_mat.coeff(cnt);
            datamat_3d[i][j][k] = data_mat.coeff(cnt);
            weightmat_3d[i][j][k] = weight_mat.coeff(cnt);
                cnt++;
            }

    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims[3] = {tx,ty, tz};
    cout << "saving matlab mat" << endl;
    cout << (save_filepath) << endl;
    //mat = Mat_Open("ttt.mat",MAT_ACC_RDWR);
    mat = Mat_Open((save_filepath).c_str(),MAT_ACC_RDWR);
    if (!mat)
    {
        mat = Mat_Create((save_filepath).c_str(), NULL);
        CHECK_NOTNULL(mat);
    }
    {
        cout << "opened mat" << endl;
        /* first matrix */
        matvar = Mat_VarCreate((string("tsdf") + var_suffix).c_str(), MAT_C_DOUBLE,MAT_T_DOUBLE,3,
                               dims, datamat_3d,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        /* secon matrix */
        matvar = Mat_VarCreate((string("weight") + var_suffix).c_str(),MAT_C_DOUBLE,MAT_T_DOUBLE,3,
                               dims,weightmat_3d,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return true;
    }
}

bool Write3DArrayMatlab(const std::vector<float>& data,
                        const Eigen::Vector3i& bbsize, const string& varname, const string& save_filepath)
{
    const int tx = bbsize[0];
    const int ty = bbsize[1];
    const int tz = bbsize[2];
    double datamat_3d[bbsize[0]][bbsize[1]][bbsize[2]];
    int cnt = 0;
    for(int i=0;i<tx;i++)
        for(int j=0;j<ty;j++)
            for (int k=0;k<tz;k++)
            {
                // to ensure matlab reads as x, y, z order
                // datamat_3d[k][j][i] = data[cnt];
                datamat_3d[i][j][k] = data[cnt];
                cnt++;
            }

    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims[3] = {tx,ty, tz};
    cout << "saving matlab mat" << endl;
    cout << (save_filepath) << endl;
    mat = Mat_Open((save_filepath).c_str(),MAT_ACC_RDWR);
    if (!mat)
    {
        mat = Mat_Create((save_filepath).c_str(), NULL);
    }
 //   if(mat)
    {
        cout << "opened mat" << endl;
        /* first matrix */
        matvar = Mat_VarCreate(varname.c_str(),MAT_C_DOUBLE,MAT_T_DOUBLE,3,
                               dims, datamat_3d,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return true;
    }
}

bool Write3DArrayMatlab(const Eigen::VectorXf &data,
                        const Eigen::Vector3i& bbsize, const string& varname, const string& tsave_filepath)
{
    string save_filepath = tsave_filepath + ".mat";
    const int tx = bbsize[0];
    const int ty = bbsize[1];
    const int tz = bbsize[2];
    double datamat_3d[bbsize[0]][bbsize[1]][bbsize[2]];
    int cnt = 0;
    for(int i=0;i<tx;i++)
        for(int j=0;j<ty;j++)
            for (int k=0;k<tz;k++)
            {
                // to ensure matlab reads as x, y, z order
                // datamat_3d[k][j][i] = data.coeff(cnt);
                datamat_3d[i][j][k] = data.coeff(cnt);
                cnt++;
            }

    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims[3] = {tx,ty, tz};
    cout << "saving matlab mat" << endl;
    cout << (save_filepath) << endl;
    mat = Mat_Open((save_filepath).c_str(),MAT_ACC_RDWR);
    if (!mat)
    {
        mat = Mat_Create((save_filepath).c_str(), NULL);
    }
 //   if(mat)
    {
        cout << "opened mat" << endl;
        /* first matrix */
        matvar = Mat_VarCreate(varname.c_str(),MAT_C_DOUBLE,MAT_T_DOUBLE,3,
                               dims, datamat_3d,0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return true;
    }
}

void ConvertCopyMatlab3DArrayToVector(std::vector<double>* vec, const double* matlab_array, const Eigen::Vector3i& bbsize)
{
    const int tx = bbsize[0];
    const int ty = bbsize[1];
    const int tz = bbsize[2];
    vec->resize(tx * ty * tz);
    int cnt = 0;
    for(int i=0;i<tx;i++)
        for(int j=0;j<ty;j++)
            for(int k=0;k<tz;k++)
            {
                (*vec)[(i * ty + j) * tz + k] = matlab_array[cnt];
                cnt++;
            }
}

void ConvertCopyMatlabMatrixToEigenVector(Eigen::MatrixXf* mat, const double* matlab_array, const Eigen::Vector2i& bbsize)
{
    const int ty = bbsize[0];
    const int tx = bbsize[1];
    mat->resize(ty,  tx);
    int cnt = 0;
    if (mat->IsRowMajor)
    {
        for(int i=0;i<tx;i++)
            for(int j=0;j<ty;j++)
            {
                (*mat)( j * tx + i ) = matlab_array[cnt];
                cnt++;
            }
    }
    else
    {
        for(int i=0;i<tx;i++)
            for(int j=0;j<ty;j++)
            {
                (*mat)( i * ty + j ) = matlab_array[cnt];
                cnt++;
            }
    }
}

void ConvertCopyMatlab3DArrayToEigenVector(Eigen::VectorXf* vec, const double* matlab_array, const Eigen::Vector3i& bbsize)
{
    const int tx = bbsize[0];
    const int ty = bbsize[1];
    const int tz = bbsize[2];
    vec->resize(tx * ty * tz);
    int cnt = 0;
    for(int i=0;i<tx;i++)
        for(int j=0;j<ty;j++)
            for(int k=0;k<tz;k++)
            {
                (*vec)[(i * ty + j) * tz + k] = matlab_array[cnt];
                cnt++;
            }
}

// append to data
bool Read3DArrayMatlab(const string &filepath, const string &varname, Eigen::VectorXf *data)
{
    vector<double> data_vec;
    Eigen::Vector3i bb_size;
    mat_t *matfp;
    matvar_t *matvar;
    matfp = Mat_Open(filepath.c_str(),MAT_ACC_RDONLY);
    if ( NULL == matfp )
    {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n",filepath.c_str());
        return false;
    }
    matvar = Mat_VarRead(matfp, varname.c_str());
    if ( NULL == matvar )
    {
        fprintf(stderr,"Variable ’%s’ not found, or error "
                       "reading MAT file\n", varname.c_str());
        return false;
    }
    else
    {
        if ( matvar->rank != 3 || matvar->isComplex || matvar->data_type != MAT_T_DOUBLE )
        {
            fprintf(stderr,"Variable ’%s’ is not a valid 3D double Array!\n", varname.c_str());
            return false;
        }
        bb_size[0] = matvar->dims[0];
        bb_size[1] = matvar->dims[1];
        bb_size[2] = matvar->dims[2];
        ConvertCopyMatlab3DArrayToEigenVector(data, (double*)(matvar->data), bb_size);
        Mat_VarFree(matvar);
    }
    Mat_Close(matfp);
    cout << "matlab mat '" << varname << "' read in finished. " << endl;
    return true;
}

bool Read3DArraysMatlab(const string &filepath, const string& filter, const Eigen::Vector3i& bbsize, vector<Eigen::VectorXf> *datas)
{
    mat_t *matfp;
    matvar_t *matvar;
    matfp = Mat_Open(filepath.c_str(),MAT_ACC_RDONLY);
    if ( NULL == matfp )
    {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n",filepath.c_str());
        return false;
    }
    while ( (matvar = Mat_VarReadNextInfo(matfp)) != NULL )
    {
        if (matvar->isComplex || matvar->rank != 3 || matvar->dims[0] != bbsize[0] ||
                matvar->dims[1] != bbsize[1] || matvar->dims[2] != bbsize[2] )
        {
            ;
        }
        else if (!filter.empty() && strstr(matvar->name, filter.c_str()) == NULL)
        {
            ;
        }
        else
        {
            printf("Reading1 %s\n",matvar->name);
            int res = Mat_VarReadDataAll(matfp, matvar);
            CHECK_EQ(res, 0);
            // printf("Reading2 %s\n",matvar->name);
            Eigen::VectorXf cur_sample;
            ConvertCopyMatlab3DArrayToEigenVector(&cur_sample, (double*)matvar->data, bbsize);
            datas->push_back(std::move(cur_sample));
        }
        Mat_VarFree(matvar);
        matvar = NULL;
    }
    return true;
}

bool ReadMatrixMatlab(const string &filepath, const string &varname, Eigen::MatrixXf *matrix)
{
    Eigen::Vector2i bb_size;
    mat_t *matfp;
    matvar_t *matvar;
    matfp = Mat_Open(filepath.c_str(),MAT_ACC_RDONLY);
    if ( NULL == matfp )
    {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n",filepath.c_str());
        return false;
    }
    matvar = Mat_VarRead(matfp, varname.c_str());
    if ( NULL == matvar )
    {
        fprintf(stderr,"Variable ’%s’ not found, or error "
                       "reading MAT file\n", varname.c_str());
        return false;
    }
    else
    {
        if ( matvar->rank != 2 || matvar->isComplex || matvar->data_type != MAT_T_DOUBLE )
        {
            fprintf(stderr,"Variable ’%s’ is not a valid double matrix. Rank: %d, iscomplex: %d, type: %d\n", varname.c_str(), matvar->rank, int(matvar->isComplex), matvar->data_type );
            fprintf(stderr, "dim y, x: %d %d\n", matvar->dims[0], matvar->dims[1]);
            return false;
        }
        bb_size[0] = matvar->dims[0];
        bb_size[1] = matvar->dims[1];
        ConvertCopyMatlabMatrixToEigenVector(matrix, (double*)matvar->data, bb_size);
        Mat_VarFree(matvar);
    }
    Mat_Close(matfp);
    cout << "matlab mat '" << varname << "' read in finished. " << endl;
    return true;
}

bool WriteMatrixMatlab(const string &save_filepath, const string &varname, const Eigen::MatrixXf &matrix)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_mat = matrix.cast<double>();
    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims[2] = {col_mat.rows(), col_mat.cols()};
    cout << "saving matlab mat" << endl;
    cout << (save_filepath) << endl;
    mat = Mat_Open((save_filepath).c_str(),MAT_ACC_RDWR);
    if (!mat)
    {
        mat = Mat_Create((save_filepath).c_str(), NULL);
    }
    {
        cout << "opened mat" << endl;
        /* first matrix */
        matvar = Mat_VarCreate(varname.c_str(),MAT_C_DOUBLE,MAT_T_DOUBLE,2,
                               dims, (double*)col_mat.data(),0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return true;
    }
}

bool WriteOBBsAndTSDFs(const TSDFHashing &scene_tsdf, const std::vector<tsdf_utility::OrientedBoundingBox> &obbs, const TSDFGridInfo &tsdf_info, const string &save_path, bool save_text_data)
{
    const int sample_num = obbs.size();
    Eigen::SparseMatrix<float, Eigen::ColMajor> samples;
    Eigen::SparseMatrix<float, Eigen::ColMajor> weights;
    ExtractSamplesFromOBBs(
            scene_tsdf,
            obbs,
            tsdf_info.boundingbox_size(),
            tsdf_info.min_model_weight(),
            &samples,
            &weights);

    for (int i = 0; i < sample_num; ++i)
    {
        const bfs::path prefix(save_path);
        std::string cur_save_path = (prefix.parent_path()/prefix.stem()).string() + "_obb_" + boost::lexical_cast<string>(i) + ".ply";
        // 1. save obb
        obbs[i].WriteToPly(cur_save_path);
        // 2. save TSDF
        TSDFHashing::Ptr cur_tsdf(new TSDFHashing);
        ConvertDataVectorToTSDFWithWeight(
        samples.col(i),
        weights.col(i),
        tsdf_info,
        cur_tsdf.get());
        string save_path = (prefix.parent_path()/prefix.stem()).string() + "_affinetrans_" + boost::lexical_cast<string>(i) + ".ply";
        TSDFHashing::Ptr transformed_tsdf(new TSDFHashing);
        float voxel_len = (scene_tsdf.voxel_length());
        TransformTSDF(*cur_tsdf, obbs[i].AffineTransform(), transformed_tsdf.get(), &voxel_len);
        WriteTSDFModel(transformed_tsdf, save_path, false, true, tsdf_info.min_model_weight());

        string save_path_canonical = (prefix.parent_path()/prefix.stem()).string() + "_canonical_" + boost::lexical_cast<string>(i) + ".ply";
        WriteTSDFModel(cur_tsdf, save_path_canonical, false, true, tsdf_info.min_model_weight());

        // 3. save data vector, weight vector, data world coordinate (all in canonical positions)
        if (save_text_data)
        {
            string text_save_path = (prefix.parent_path()/prefix.stem()).string() + "_textdata_canonical_" + boost::lexical_cast<string>(i) + ".txt";
            Eigen::MatrixXf save_mat(samples.rows(), 8);
            save_mat.setZero();
            save_mat.col(0) = samples.col(i); // data
            save_mat.col(1) = weights.col(i); // weight

            TSDFHashing tmp_tsdf;
            std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f>> idx_worldpos;
            ConvertDataVectorToTSDFWithWeightAndWorldPos(samples.col(i),
                                                         weights.col(i),
                                                         tsdf_info,
                                                         &tmp_tsdf,
                                                         &idx_worldpos);
            for (std::map<int, std::pair<Eigen::Vector3i, Eigen::Vector3f>>::const_iterator citr = idx_worldpos.begin(); citr !=  idx_worldpos.end(); ++citr)
            {
                CHECK_LT(citr->first, save_mat.rows());
                save_mat.coeffRef(citr->first, 2 + 0) = (citr->second).first[0];
                save_mat.coeffRef(citr->first, 2 + 1) = (citr->second).first[1];
                save_mat.coeffRef(citr->first, 2 + 2) = (citr->second).first[2];

                save_mat.coeffRef(citr->first, 2 + 3) = (citr->second).second[0];
                save_mat.coeffRef(citr->first, 2 + 4) = (citr->second).second[1];
                save_mat.coeffRef(citr->first, 2 + 5) = (citr->second).second[2];
            }
            utility::WriteEigenMatrix(save_mat, text_save_path);
        }
    }
    return true;
}

} // end namespace cpu_tsdf
