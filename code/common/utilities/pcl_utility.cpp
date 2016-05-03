#include "pcl_utility.h"
#include <algorithm>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/search/kdtree.h>
#include <glog/logging.h>

namespace utility {
void
flattenVerticesNoRGB (pcl::PolygonMesh &mesh, float min_dist /*= 0.00005*/)
{
  // Remove duplicated vertices
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
  pcl::search::KdTree<pcl::PointXYZ> vert_tree (true);
  vert_tree.setInputCloud (vertices);
  // Find duplicates
  std::vector<int> vertex_remap (vertices->size (), -1);
  int idx = 0;
  std::vector<int> neighbors;
  std::vector<float> dists;
  pcl::PointCloud<pcl::PointXYZ> vertices_new;
  //cv::Vec3f average_color = cv::Vec3f();
  //pcl::PointXYZ nan_point(NAN, NAN, NAN);
  for (size_t i = 0; i < vertices->size (); i++)
  {
    const pcl::PointXYZ& cur_pt = vertices->at(i);
    if ( isnan(cur_pt.x) || isnan(cur_pt.y) || isnan(cur_pt.z))
    {
        continue;
    }
    if (vertex_remap[i] >= 0)
      continue;
    vertex_remap[i] = idx;
    vert_tree.radiusSearch (i, min_dist, neighbors, dists);
    if (neighbors.empty())
        continue;
    //assert(i == neighbors[0]);

    //average_color = cv::Vec3f(cur_pt.r, cur_pt.g, cur_pt.b);
    for (size_t j = 1; j < neighbors.size (); j++)
    {
      if (dists[j] < min_dist)
      {
        vertex_remap[neighbors[j]] = idx;
        const pcl::PointXYZ& cur_pt = vertices->at(neighbors[j]);
        //average_color[0] += cur_pt.r;
        //average_color[1] += cur_pt.g;
        //average_color[2] += cur_pt.b;
      }
    }
    pcl::PointXYZ& pt_final = vertices->at(i);
//    pt_final.r = static_cast<uchar>(average_color[0]/neighbors.size());
//    pt_final.g = static_cast<uchar>(average_color[1]/neighbors.size());
//    pt_final.b = static_cast<uchar>(average_color[2]/neighbors.size());
    vertices_new.push_back (pt_final);
    idx++;
  }
  std::vector<size_t> faces_to_remove;
  size_t face_idx = 0;
  for (size_t i = 0; i < mesh.polygons.size (); i++)
  {
    pcl::Vertices &v = mesh.polygons[i];
    for (size_t j = 0; j < v.vertices.size (); j++)
    {
      v.vertices[j] = vertex_remap[v.vertices[j]];
    }
    if (v.vertices[0] == v.vertices[1] || v.vertices[1] == v.vertices[2] || v.vertices[2] == v.vertices[0])
    {
      PCL_INFO ("Degenerate face: (%d, %d, %d)\n", v.vertices[0], v.vertices[1], v.vertices[2]);
    }
    else
    {
      mesh.polygons[face_idx++] = mesh.polygons[i];
    }
  }
  mesh.polygons.resize (face_idx);
  pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
}

void
flattenVertices (pcl::PolygonMesh &mesh, float min_dist /*= 0.00005*/)
{
    CHECK_GT(min_dist, 0);
  // Remove duplicated vertices
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
  pcl::search::KdTree<pcl::PointXYZRGB> vert_tree (true);
  vert_tree.setInputCloud (vertices);
  // Find duplicates
  std::vector<int> vertex_remap (vertices->size (), -1);
  int idx = 0;
  std::vector<int> neighbors;
  std::vector<float> sq_dists;
  pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
  cv::Vec3f average_color = cv::Vec3f();
  for (size_t i = 0; i < vertices->size (); i++)
  {
    const pcl::PointXYZRGB& cur_pt = vertices->at(i);
    if (vertex_remap[i] >= 0)
      continue;
    if ( isnan(cur_pt.x) || isnan(cur_pt.y) || isnan(cur_pt.z))
    {
        continue;
    }
    vertex_remap[i] = idx;
    vert_tree.radiusSearch (i, min_dist, neighbors, sq_dists);
    if (neighbors.empty())
        continue;
    //assert(i == neighbors[0]);
    //average_color = cv::Vec3f(cur_pt.r, cur_pt.g, cur_pt.b);
    average_color = cv::Vec3f();
    cv::Vec3f average_pos = cv::Vec3f();
    for (size_t j = 0; j < neighbors.size (); j++)
    {
      // if (sq_dists[j] < min_dist)
      {
        vertex_remap[neighbors[j]] = idx;
        const pcl::PointXYZRGB& cur_pt = vertices->at(neighbors[j]);
        average_color[0] += cur_pt.r;
        average_color[1] += cur_pt.g;
        average_color[2] += cur_pt.b;
        average_pos[0] += cur_pt.x;
        average_pos[1] += cur_pt.y;
        average_pos[2] += cur_pt.z;
      }
    }
    pcl::PointXYZRGB& pt_final = vertices->at(i);
    pt_final.r = static_cast<uchar>(average_color[0]/neighbors.size());
    pt_final.g = static_cast<uchar>(average_color[1]/neighbors.size());
    pt_final.b = static_cast<uchar>(average_color[2]/neighbors.size());
    pt_final.x = average_pos[0]/neighbors.size();
    pt_final.y = average_pos[1]/neighbors.size();
    pt_final.z = average_pos[2]/neighbors.size();
    vertices_new.push_back (pt_final);
    idx++;
  }
  std::vector<size_t> faces_to_remove;
  size_t face_idx = 0;
  for (size_t i = 0; i < mesh.polygons.size (); i++)
  {
    pcl::Vertices &v = mesh.polygons[i];
    for (size_t j = 0; j < v.vertices.size (); j++)
    {
      v.vertices[j] = vertex_remap[v.vertices[j]];
      CHECK_GE(v.vertices[j], 0);
    }
    if (v.vertices[0] == v.vertices[1] || v.vertices[1] == v.vertices[2] || v.vertices[2] == v.vertices[0])
    {
      //PCL_INFO ("Degenerate face: (%d, %d, %d)\n", v.vertices[0], v.vertices[1], v.vertices[2]);
    }
    else
    {
      mesh.polygons[face_idx++] = mesh.polygons[i];
    }
  }
  mesh.polygons.resize (face_idx);
  pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
}

pcl::PointCloud<pcl::PointNormal>::Ptr
meshToFaceCloud (const pcl::PolygonMesh &mesh)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ> vertices;
  pcl::fromPCLPointCloud2 (mesh.cloud, vertices);

  for (size_t i = 0; i < mesh.polygons.size (); ++i)
  {
    if (mesh.polygons[i].vertices.size () != 3)
    {
      PCL_ERROR ("Found a polygon of size %d\n", mesh.polygons[i].vertices.size ());
      continue;
    }
    Eigen::Vector3f v0 = vertices.at (mesh.polygons[i].vertices[0]).getVector3fMap ();
    Eigen::Vector3f v1 = vertices.at (mesh.polygons[i].vertices[1]).getVector3fMap ();
    Eigen::Vector3f v2 = vertices.at (mesh.polygons[i].vertices[2]).getVector3fMap ();
    float area = ((v1 - v0).cross (v2 - v0)).norm () / 2. * 1E4;
    Eigen::Vector3f normal = ((v1 - v0).cross (v2 - v0));
    normal.normalize ();
    pcl::PointNormal p_new;
    p_new.getVector3fMap () = (v0 + v1 + v2)/3.;
    p_new.normal_x = normal (0);
    p_new.normal_y = normal (1);
    p_new.normal_z = normal (2);
    cloud->points.push_back (p_new);
  }
  cloud->height = 1;
  cloud->width = cloud->size ();
  return (cloud);
}

void
cleanupMesh (pcl::PolygonMesh &mesh, float face_dist/*=0.02*/, int min_neighbors/*=5*/)
{
  // Remove faces which aren't within 2 marching cube widths from any others
  pcl::PointCloud<pcl::PointNormal>::Ptr faces = meshToFaceCloud (mesh);
  std::vector<size_t> faces_to_remove;
  pcl::search::KdTree<pcl::PointNormal>::Ptr face_tree (new pcl::search::KdTree<pcl::PointNormal>);
  face_tree->setInputCloud (faces);
  // Find small clusters and remove them
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointNormal> extractor;
  extractor.setInputCloud (faces);
  extractor.setSearchMethod (face_tree);
  extractor.setClusterTolerance (face_dist);
  extractor.setMaxClusterSize(min_neighbors);
  extractor.extract(clusters);
  PCL_INFO ("Found %d clusters\n", clusters.size ());
  // Aggregate indices
  std::vector<bool> keep_face (faces->size (), false);
  for(size_t i = 0; i < clusters.size(); i++)
  {
    for(size_t j = 0; j < clusters[i].indices.size(); j++)
    {
      faces_to_remove.push_back (clusters[i].indices[j]);
    }
  }
  std::sort (faces_to_remove.begin (), faces_to_remove.end ());
  // Remove the face
  for (int64_t i = faces_to_remove.size () - 1; i >= 0; i--)
  {
    mesh.polygons.erase (mesh.polygons.begin () + faces_to_remove[i]);
  }
  // Remove all vertices with no face
  pcl::PointCloud<pcl::PointXYZ> vertices;
  pcl::fromPCLPointCloud2 (mesh.cloud, vertices);
  std::vector<bool> has_face (vertices.size (), false);
  for (size_t i = 0; i < mesh.polygons.size (); i++)
  {
    const pcl::Vertices& v = mesh.polygons[i];
    has_face[v.vertices[0]] = true;
    has_face[v.vertices[1]] = true;
    has_face[v.vertices[2]] = true;
  }
  pcl::PointCloud<pcl::PointXYZ> vertices_new;
  std::vector<size_t> get_new_idx (vertices.size ());
  size_t cur_idx = 0;
  for (size_t i = 0; i <vertices.size (); i++)
  {
    if (has_face[i])
    {
      vertices_new.push_back (vertices[i]);
      get_new_idx[i] = cur_idx++;
    }
  }
  for (size_t i = 0; i < mesh.polygons.size (); i++)
  {
    pcl::Vertices &v = mesh.polygons[i];
    v.vertices[0] = get_new_idx[v.vertices[0]];
    v.vertices[1] = get_new_idx[v.vertices[1]];
    v.vertices[2] = get_new_idx[v.vertices[2]];
  }
  pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
}

void Write3DPointToFilePCL(const std::string &fname, const std::vector<cv::Vec3d> &points3d, const std::vector<cv::Vec3b> *colors)
{
    //      FILE* hf = fopen(fname.c_str(), "w");
    //      assert(hf);
    //      for(int i=0; i < points3d.size(); ++i) {
    //          fprintf(hf, "%f %f %f", points3d[i][0], points3d[i][1], points3d[i][2]);
    //          if (colors)
    //          {
    //              fprintf(hf, " %d %d %d\n", (*colors)[i][0], (*colors)[i][1], (*colors)[i][2]);
    //          }
    //          else
    //          {
    //              fprintf(hf, "\n");
    //          }
    //      }
    //      fclose(hf);

    pcl::PointCloud<pcl::PointXYZRGB> pcl;
    for (int tt = 0; tt < points3d.size(); ++tt)
    {
        pcl::PointXYZRGB curpt;
        curpt.x = points3d[tt][0];
        curpt.y = points3d[tt][1];
        curpt.z = points3d[tt][2];
        curpt.r = (*colors)[tt][0];
        curpt.g = (*colors)[tt][1];
        curpt.b = (*colors)[tt][2];
        pcl.push_back(curpt);
    }
    pcl::io::savePLYFileBinary (fname, pcl);
}

int ComputeRemap(const std::vector<bool> &kept_verts, std::vector<int>* remap)
{
    remap->resize(kept_verts.size());
    int cur_valid_idx = 0;
    for (int i = 0; i < kept_verts.size(); ++i)
    {
        if (kept_verts[i])
        {
            (*remap)[i] = cur_valid_idx;
            cur_valid_idx++;
        }
        else
        {
            (*remap)[i] = -1;
        }
    }
    return cur_valid_idx;
}

void ClearMeshWithVertKeepArray(pcl::PolygonMesh &mesh, const std::vector<bool> &kept_verts)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
    cout << "before " << kept_verts.size() << endl;
    CHECK_EQ(vertices->size(), kept_verts.size());
    std::vector<int> remap;
    int tot_kept_verts = ComputeRemap(kept_verts, &remap);
    for (int i = 0; i < mesh.polygons.size(); ++i)
    {
        pcl::Vertices& cur_tri = mesh.polygons[i];
        CHECK_EQ(cur_tri.vertices.size(), 3);
        cur_tri.vertices[0] = remap[cur_tri.vertices[0]];
        cur_tri.vertices[1] = remap[cur_tri.vertices[1]];
        cur_tri.vertices[2] = remap[cur_tri.vertices[2]];
    }
    static const uint32_t invalid_idx = 0xffffffff;
    mesh.polygons.erase(std::remove_if(mesh.polygons.begin(), mesh.polygons.end(), [](const pcl::Vertices& tri){
        return tri.vertices[0] == invalid_idx || tri.vertices[1]  == invalid_idx|| tri.vertices[2] == invalid_idx;
    }), mesh.polygons.end());
    pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
    int cnt = 0;
    for (int i = 0; i < vertices->size(); ++i)
    {
        if (kept_verts[i])
        {
            vertices_new.push_back(vertices->at(i));
            cnt++;
        }
    }
    CHECK_EQ(cnt, tot_kept_verts);
    cout << "cur: "<< cnt << endl;
    pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
}

//void ReservedVertToReservedTris(pcl::PolygonMesh &mesh,
//                                const std::vector<bool> &kept_verts,
//                                const std::vector<bool)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices (new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::fromPCLPointCloud2 (mesh.cloud, *vertices);
//    CHECK_EQ(vertices->size(), kept_verts.size());
//    std::vector<int> remap;
//    int tot_kept_verts = ComputeRemap(kept_verts, &remap);
//    for (int i = 0; i < mesh.polygons.size(); ++i)
//    {
//        pcl::Vertices& cur_tri = mesh.polygons[i];
//        CHECK_EQ(cur_tri.vertices.size(), 3);
//        cur_tri.vertices[0] = remap[cur_tri.vertices[0]];
//        cur_tri.vertices[1] = remap[cur_tri.vertices[1]];
//        cur_tri.vertices[2] = remap[cur_tri.vertices[2]];
//    }
//    mesh.polygons.erase(std::remove_if(mesh.polygons.begin(), mesh.polygons.end(), [](const pcl::Vertices& tri){
//        return tri.vertices[0] < 0 || tri.vertices[1] < 0 || tri.vertices[2] < 0;
//    }), mesh.polygons.end());
//    pcl::PointCloud<pcl::PointXYZRGB> vertices_new;
//    int cnt = 0;
//    for (int i = 0; i < vertices->size(); ++i)
//    {
//        if (kept_verts[i])
//        {
//            vertices_new.push_back(vertices->at(i));
//        }
//    }
//    CHECK_EQ(cnt, tot_kept_verts);
//    pcl::toPCLPointCloud2 (vertices_new, mesh.cloud);
//}

}
