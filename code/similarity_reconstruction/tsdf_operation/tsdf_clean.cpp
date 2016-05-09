/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "tsdf_clean.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/functional/hash/extensions.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <glog/logging.h>

#include "common/utilities/pcl_utility.h"
#include "common/utilities/eigen_utility.h"
#include "tsdf_representation/tsdf_hash.h"
#include "marching_cubes/marching_cubes_tsdf_hash.h"
#include "tsdf_hash_utilities/utility.h"
#include "tsdf_operation/tsdf_io.h"

using namespace std;

namespace  {
struct NeighborTriIdx
{
    int t1;
    int t2;
    NeighborTriIdx():t1(-1),t2(-1){}
    NeighborTriIdx(int vt1, int vt2):t1(vt1), t2(vt2){}
};

struct pair_hash
    : std::unary_function<std::pair<int, int>, std::size_t>
{
    std::size_t operator()(const std::pair<int, int> & x) const
    {
        std::size_t seed = 0;
        return boost::hash_value(x);
    }
};

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, boost::no_property, boost::no_property, boost::no_property, boost::vecS> Graph;
typedef std::unordered_map<std::pair<int, int>,  NeighborTriIdx, pair_hash> EdgeMap;


std::pair<int, int> MakeAscendPair(int v1, int v2)
{
    if (v1 > v2) swap(v1, v2);
    return make_pair(v1, v2);
}

bool MeshToTriangleEdgeMap(const pcl::PolygonMesh &mesh,  EdgeMap *tri_edge_map)
{
    for (size_t i = 0; i < mesh.polygons.size (); i++)
    {
        const pcl::Vertices &v = mesh.polygons[i];
        pair<int, int> edges[3];
        edges[0] = MakeAscendPair(v.vertices[0], v.vertices[1]);
        edges[1] = MakeAscendPair(v.vertices[1], v.vertices[2]);
        edges[2] = MakeAscendPair(v.vertices[2], v.vertices[0]);
        for (int j = 0; j < 3; ++j)
        {
            const auto& edge = edges[j];
            CHECK_LE(edge.first, edge.second);
            auto itr = tri_edge_map->find(edge);
            if (itr == tri_edge_map->end())
            {
                tri_edge_map->insert(make_pair(edge, NeighborTriIdx(i, -1)));
            }
            else
            {
                CHECK_GT(itr->second.t1, -1);
                itr->second.t2 = i;
            }
        }
    }
    return true;
}

bool TriangleEdgeMapToBoostGraph(
        const EdgeMap &tri_edge_map,
        Graph* graph)
{
        // convert to boost::graph
        using namespace boost;
        Graph& G = *graph;
        // build graph
        std::cout << "begin building graph" << endl;
        G.m_edges.reserve(tri_edge_map.size());
        for(auto& pair : tri_edge_map)
        {
            if (pair.second.t2 > 0)
            {
                CHECK_GT(pair.second.t1, -1);
                add_edge(pair.second.t1,pair.second.t2, G);
            }
            else
            {
                CHECK_GT(pair.second.t1, -1);
                add_edge(pair.second.t1,pair.second.t1, G);
                //add_vertex(pair.second.t1, G);
            }
        }
        std::cout << "Building graph done" << endl;
        return true;
}

int LargestConnectedComponent(const Graph& G)
{
    using namespace boost;
    // find connected components
    std::vector<int> component(num_vertices(G));
    int num_conn = connected_components(G, &component[0]);
    return num_conn;
}

vector<int> ComponentSizes(const int comp_num, const std::vector<int>& connected_components)
{
    vector<int> component_sizes(comp_num, 0);
    for (int i = 0; i < connected_components.size(); ++i)
    {
        component_sizes[connected_components[i]]++;
    }
    return component_sizes;
}

bool FilterTriangle(const std::vector<int>& connected_components, const int comp_num, int thresh, std::vector<bool>* reserved_tri)
{
    if (comp_num == 0) return false;
    vector<int> component_sizes = ComponentSizes(comp_num, connected_components);
    // utility::OutputVector("/home/dell/debug_compnun.txt", component_sizes);
    if (thresh < 0)
    {
        thresh = *(std::max_element(component_sizes.begin(), component_sizes.end()));
    }
    reserved_tri->resize(connected_components.size(), true);
    for (int i = 0; i < connected_components.size(); ++i)
    {
        if (component_sizes[connected_components[i]] < thresh)
        {
            (*reserved_tri)[i] = false;
        }
    }
    return true;
}

}

namespace cpu_tsdf {
bool CleanMesh(
        pcl::PolygonMesh& mesh,
        float compo_thresh
        )
{
    CHECK_GE(mesh.polygons.size(), 0);
    utility::flattenVertices(mesh);

    EdgeMap edge_map;
    MeshToTriangleEdgeMap(mesh, &edge_map);
    // std::cout << edge_map.size() << std::endl;
    Graph G;
    TriangleEdgeMapToBoostGraph(edge_map, &G);
    edge_map.clear();
    std::cout << num_vertices(G) << std::endl;
    std::vector<int> component(num_vertices(G));

    int num_conn = connected_components(G, &component[0]);
    // std::cout << "conn num: " << num_conn << std::endl;
    G.clear();
    std::vector<bool> reserved_tri;
    FilterTriangle(component, num_conn, compo_thresh, &reserved_tri);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
    std::vector<bool> reserved_verts(vertices->size(), true);
    for (int i = 0; i < reserved_tri.size(); ++i)
    {
        if (reserved_tri[i] == false)
        {
            const ::pcl::Vertices& cur_tri = mesh.polygons[i];
            reserved_verts[cur_tri.vertices[0]] = false;
            reserved_verts[cur_tri.vertices[1]] = false;
            reserved_verts[cur_tri.vertices[2]] = false;
        }
    }
    utility::ClearMeshWithVertKeepArray(mesh, reserved_verts);
}

bool ComputeTSDFCleanVoxels(
        const cpu_tsdf::TSDFHashing::Ptr tsdf,
        int compo_thresh,
        std::vector<Eigen::Vector3i> *voxel_to_remove,
        int neighborst,
        int neighbored
        )
{
    pcl::PolygonMesh::Ptr mesh = cpu_tsdf::TSDFToPolygonMesh(tsdf, 0);
    CHECK_GE(mesh->polygons.size(), 0);
    EdgeMap edge_map;
    MeshToTriangleEdgeMap(*mesh, &edge_map);
    // std::cout << edge_map.size() << std::endl;
    Graph G;
    TriangleEdgeMapToBoostGraph(edge_map, &G);
    edge_map.clear();
    // std::cout << num_vertices(G) << std::endl;
    std::vector<int> component(num_vertices(G));
    int num_conn = connected_components(G, &component[0]);
    // std::cout << "conn num: " << num_conn << std::endl;
    G.clear();
    std::vector<bool> reserved_tri;
    FilterTriangle(component, num_conn, compo_thresh, &reserved_tri);
    component.clear();
    ComputeRemovedVoxelsFromRemovedTris(reserved_tri, *mesh, tsdf.get(), voxel_to_remove, neighborst, neighbored);
    return true;
}

void CleanTSDFVoxels(const std::vector<Eigen::Vector3i>& voxel_to_remove, cpu_tsdf::TSDFHashing::Ptr tsdf)
{
    for (int i = 0; i < voxel_to_remove.size(); ++i)
    {
        tsdf->SetTSDFValue(utility::EigenVectorToCvVector3(voxel_to_remove[i]), 0, 0, cv::Vec3b(0, 0, 0));
    }
}

void CleanTSDF(
        cpu_tsdf::TSDFHashing::Ptr tsdf,
        int compo_thresh,
        int neighborst,
        int neighbored
        )
{
    std::vector<Eigen::Vector3i> voxel_to_remove;
    ComputeTSDFCleanVoxels(tsdf, compo_thresh, &voxel_to_remove, neighborst, neighbored);
    CleanTSDFVoxels(voxel_to_remove, tsdf);
}

void CleanTSDFs(
        std::vector<cpu_tsdf::TSDFHashing::Ptr>& tsdfs,
        int compo_thresh,
        int neighborst,
        int neighbored
        )
{
    for (int i  = 0; i < tsdfs.size(); ++i) {
        CleanTSDF(tsdfs[i], compo_thresh, neighborst, neighbored);
    }
}

void CleanTSDFSampleVector(const cpu_tsdf::TSDFHashing& scene_tsdf,
        const Eigen::SparseVector<float> &sample,
        const Eigen::Vector3i boundingbox_size,
        const int compo_thresh,
        Eigen::SparseVector<float> *weight,
                           int neighborst,
                           int neighbored
        )
{
    const int total_size = boundingbox_size[0] * boundingbox_size[1] * boundingbox_size[2];
    CHECK_EQ(total_size, weight->size());
    cpu_tsdf::TSDFGridInfo grid_info(scene_tsdf, boundingbox_size, 0);
    cpu_tsdf::TSDFHashing::Ptr cur_sample(new cpu_tsdf::TSDFHashing);
    cpu_tsdf::ConvertDataVectorToTSDFWithWeight(sample, *weight, grid_info, cur_sample.get());
    std::vector<Eigen::Vector3i> voxel_to_remove;
    ComputeTSDFCleanVoxels(cur_sample, compo_thresh, &voxel_to_remove, neighborst, neighbored);
    for (int i = 0; i < voxel_to_remove.size(); ++i)
    {
        int idx = cpu_tsdf::Subscript3DToIndex(boundingbox_size, voxel_to_remove[i]);
        if (idx >=0 && idx < total_size)
        {
            weight->coeffRef(idx) = 0;
        }
    }
    //weight->prune();
    return;
}

void CleanTSDFSampleMatrix(
        const TSDFHashing &scene_tsdf,
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const Eigen::Vector3i boundingbox_size,
        const int compo_thresh,
        Eigen::SparseMatrix<float, Eigen::ColMajor> *weights,
        int neighborst,
        int neighbored)
{
    for (int i = 0; i < samples.cols(); ++i)
    {
        Eigen::SparseVector<float> cur_weight = weights->col(i);
        cout << "cur_weight: " << cur_weight.nonZeros() << endl;
        CleanTSDFSampleVector(scene_tsdf, samples.col(i), boundingbox_size, compo_thresh, &cur_weight, neighborst, neighbored);
        weights->col(i) = cur_weight;
    }
}

void ComputeRemovedVoxelsFromRemovedTris(
        const std::vector<bool>& reserved_tri,
        const pcl::PolygonMesh& mesh,
        const cpu_tsdf::TSDFHashing* tsdf,
        std::vector<Eigen::Vector3i>* removed_voxels,
        int neighborst,
        int neighbored)
{
    if (reserved_tri.empty()) return;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *points);
    CHECK_EQ(reserved_tri.size(), mesh.polygons.size());
    for (int i = 0; i < reserved_tri.size(); ++i)
    {
        if (reserved_tri[i] == false)
        {
            const pcl::Vertices &v = mesh.polygons[i];
            for (int j = 0; j < v.vertices.size(); ++j)
            {
                cv::Vec3f voxel_coord = tsdf->World2Voxel( utility::PCLPoint2CvVec(points->at(v.vertices[j])) );
                cv::Vec3i voxel_coord_int = utility::round(voxel_coord);
                for (int x = neighborst; x < neighbored; ++x)
                    for (int y = neighborst; y < neighbored; ++y)
                        for (int z = neighborst; z < neighbored; ++z)
                        {
                            cv::Vec3i cur_voxel = voxel_coord_int + cv::Vec3i(x, y, z);
                            removed_voxels->push_back(utility::CvVectorToEigenVector3(cur_voxel));
                        }
            }
        }
    }
}

void ComputeRemovedVoxelsFromRemovedVerts(
        const std::vector<bool>& reserved_verts,
        const pcl::PolygonMesh& mesh,
        const cpu_tsdf::TSDFHashing* tsdf,
        std::vector<Eigen::Vector3i>* removed_voxels,
        int neighborst,
        int neighbored)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *points);
    CHECK_EQ(reserved_verts.size(), points->size());
    for (int i = 0; i < reserved_verts.size(); ++i)
    {
        if (reserved_verts[i] == false)
        {
            cv::Vec3f voxel_coord = tsdf->World2Voxel( utility::PCLPoint2CvVec(points->at(i)) );
            cv::Vec3i voxel_coord_int = utility::round(voxel_coord);
            for (int x = neighborst; x < neighbored; ++x)
                for (int y = neighborst; y < neighbored; ++y)
                    for (int z = neighborst; z < neighbored; ++z)
                    {
                        cv::Vec3i cur_voxel = voxel_coord_int + cv::Vec3i(x, y, z);
                        removed_voxels->push_back(utility::CvVectorToEigenVector3(cur_voxel));
                    }
        }
    }
}

void CleanTSDFFromMeshVerts(TSDFHashing::Ptr tsdf, const pcl::PolygonMesh &mesh, const std::vector<bool> &reserved_verts, int neighborst, int neighbored)
{
    std::vector<Eigen::Vector3i> removed_voxels;
    ComputeRemovedVoxelsFromRemovedVerts(reserved_verts, mesh, tsdf.get(), &removed_voxels, neighborst, neighbored);
    CleanTSDFVoxels(removed_voxels, tsdf);
}

bool CleanNoiseInSamples(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        const std::vector<int>& model_assign_idx,
        const std::vector<double>& outlier_gammas,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* pweights,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* valid_obs_weight_mat,
        float counter_thresh, float pos_trunc, float neg_trunc)
{
    // cout << "begin clean noise" << endl;
    Eigen::SparseMatrix<float, Eigen::ColMajor>& weights = *pweights;
    valid_obs_weight_mat->resize(pweights->rows(), pweights->cols());
    valid_obs_weight_mat->setZero();

    const int feature_dim = samples.rows();
    const int model_number = *(std::max_element(model_assign_idx.begin(), model_assign_idx.end())) + 1;
    vector<vector<int>> cluster_sample_idx;
    GetClusterSampleIdx(model_assign_idx, outlier_gammas, model_number, &cluster_sample_idx);
    for (int i = 0; i < model_number; ++i)
    {
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_samples(feature_dim, cluster_sample_idx[i].size());
        Eigen::SparseMatrix<float, Eigen::ColMajor> cur_weights(feature_dim, cluster_sample_idx[i].size());
        Eigen::SparseVector<float> cur_valid_obs(feature_dim);
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            cur_samples.col(j) = samples.col(cluster_sample_idx[i][j]);
            cur_weights.col(j) = weights.col(cluster_sample_idx[i][j]);
        }
        CleanNoiseInSamplesOneCluster(
                    cur_samples, &cur_weights, &cur_valid_obs, counter_thresh,
                    pos_trunc, neg_trunc);
        for (int j = 0; j < cluster_sample_idx[i].size(); ++j)
        {
            weights.col(cluster_sample_idx[i][j]) = cur_weights.col(j);
            valid_obs_weight_mat->col(cluster_sample_idx[i][j]) = cur_valid_obs;
        }
    }  // end for i
    weights.prune(0, 1);
    return true;
}

bool CleanNoiseInSamplesOneCluster(
        const Eigen::SparseMatrix<float, Eigen::ColMajor> &samples,
        Eigen::SparseMatrix<float, Eigen::ColMajor>* weights,
        Eigen::SparseVector<float>* valid_obs_positions,
        float counter_thresh, float pos_trunc, float neg_trunc)
{
    const float abs_neg_trunc = fabs(neg_trunc);
    const float abs_pos_trunc = fabs(pos_trunc);
    const float weight_thresh = 0.00;
    if (samples.size() == 0 || weights->size() == 0) return false;
    Eigen::VectorXf obs_counter = Eigen::VectorXf::Zero(weights->rows());
    cout << "go through mat" << endl;
    for (int k=0; k<weights->outerSize(); ++k)
      for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it(*weights,k); it; ++it)
      {
          if (it.value() > weight_thresh)
          {
              float cur_dist = samples.coeff(it.row(), it.col());
              //obs_counter(it.row()) += 1 - cur_dist/(cur_dist >= 0 ? abs_pos_trunc : abs_neg_trunc);
              obs_counter(it.row()) += 1;
          }
      }
    cout << "begin setting zeros" << endl;
    // second time
    valid_obs_positions->resize(samples.rows());
    valid_obs_positions->setZero();
    for (int k=0; k<weights->outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator it(*weights,k); it; ++it)
      {
          if (obs_counter(it.row()) < counter_thresh)
          {
              // cout << "clean " << it.row() << " " << it.col() <<endl;
              it.valueRef() = 0;
          }
      }
    }  // end k
    for (int r = 0; r < obs_counter.size(); ++r)
    {
        if (obs_counter(r) >= counter_thresh)
        {
            valid_obs_positions->coeffRef(r) = 1.0 * cpu_tsdf::TSDFHashing::getVoxelMaxWeight();
        }
    }
    // cpu_tsdf::Write3DArrayMatlab(obs_counter, options.boundingbox_size, "obs_counter", options.save_path);
    cout << "finished setting zeros" << endl;
    weights->prune(weight_thresh, 1);
    return true;
}

}
