#include "urban_reconstruction_data_load.h"

#include <string>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <opencv2/opencv.hpp>

#include "common/fisheye_camera/RectifiedCameraPair.h"

namespace bfs = boost::filesystem;
using std::string;
using std::vector;

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 std::vector<RectifiedCameraPair>* cam_infos,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files);

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 std::vector<std::string>* cam_info_files,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files);

cv::Mat LoadSemanticLabelAsMat(const std::string & semantic_label_filename)
{
  using std::cout;
  using std::endl;
  if (semantic_label_filename.empty() ||
      !bfs::exists(semantic_label_filename)) return cv::Mat();

  std::ifstream ifs(semantic_label_filename);
  string dummy;
  char dummy_char;
  string original_file;

  ifs >> dummy >> original_file;
  int imw = 0;
  int imh = 0;
  cv::Mat origin_im = cv::imread(original_file);
  imh = origin_im.rows;
  imw = origin_im.cols;
  assert(imh > 0 && imw > 0);

  cv::Mat label_mat = cv::Mat::zeros(imh, imw, CV_16UC1);
  ifs.get();
  while (ifs >> dummy) {
      assert(dummy == "object:");
      int cur_obj_cnt = 0;
      ifs >> cur_obj_cnt;

      int cur_semantic_label = -1;
      ifs >> dummy >> cur_semantic_label;
      assert(dummy == "<housenumber>:");
      assert(cur_semantic_label > 0);
      ifs.get();

      cv::Rect bbox;
      std::getline(ifs, dummy);
      sscanf(dummy.c_str(), "bbox: %d,%d,%d,%d", &bbox.x, &bbox.y, &bbox.width, &bbox.height);

      ifs.get();

      label_mat(bbox) = (unsigned short)cur_semantic_label;

      std::cout << "original_file: " << original_file << endl;
      std::cout << "cur_obj: " << cur_obj_cnt << endl;
      std::cout << "housenum: " << cur_semantic_label << endl;
      std::cout << "bbox: " << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << endl;
    }
  cv::imwrite(semantic_label_filename + ".label.png", label_mat);
  return label_mat;
}

bool ListFiles(const std::string& dir, const std::string& extension, std::vector<std::string>* filelist)
{
    if(!bfs::exists(dir))
        return false;
    bfs::directory_iterator end_itr;
    for (bfs::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        std::string cur_extension = boost::algorithm::to_lower_copy
                                    (bfs::extension (itr->path ()));
        //std::string basename = bfs::basename (itr->path ());
        std::string filename = (itr->path ().filename().string());
        std::string pathname = itr->path ().string ();
        if (cur_extension == extension && filename[0]<='9' && filename[0]>='0')
        {
            filelist->push_back(pathname);
        }
    }
    std::sort(filelist->begin(), filelist->end());
    return true;
}

bool ListDirs(const std::string &root_dir, const std::string &prefix, std::vector<std::string> *dirlist)
{
    using namespace std;
    if(!bfs::exists(root_dir))
        return false;
    bfs::directory_iterator end_itr;
    for (bfs::directory_iterator itr (root_dir); itr != end_itr; ++itr)
    {
        const string dirname = itr->path().filename().string();
        if (bfs::is_directory(*itr) && dirname.length() >= prefix.length() && std::equal(prefix.begin(), prefix.end(), dirname.begin())) {
            dirlist->push_back(itr->path().string());
        }
    }
    std::sort(dirlist->begin(), dirlist->end());
    return true;
}

bool ListFilesWithinFrameRange(const std::string& dir, const std::string& ext, int start_image, int end_image, std::vector<std::string>* filelist)
{
    using namespace bfs;
//    int total_file = std::count_if(
//                directory_iterator(dir),
//                directory_iterator(),
//                bind( static_cast<bool(*)(const path&)>(is_regular_file),
//                bind( &directory_entry::path, std::placeholders::_1 ) ) );
    start_image = std::max(start_image, 0);
    if (end_image < 0) end_image = 1e8;
    // end_image = std::min(end_image, total_file);
    if (!bfs::exists(dir)) return false;
    static const char* file_pattern_stem = "%010d_%010d";
    const string file_pattern = string(file_pattern_stem) + ext;
    char filename[128];
    for (int i = start_image; i <= end_image; ++i)
    {
        int j = i;
        bfs::path current_path;
        do
        {
            j--;
            sprintf(filename, file_pattern.c_str(), i, j);
            current_path = bfs::path(dir)/filename;
        } while (j >= 0 && !bfs::exists(current_path));
        if (bfs::exists(current_path))
        {
            filelist->push_back(current_path.string());
        }

        j = i;
        current_path.clear();
        do
        {
            j++;
            sprintf(filename, file_pattern.c_str(), i, j);
            current_path = bfs::path(dir)/filename;
        } while (j - i < 50 && !bfs::exists(current_path));
        if (bfs::exists(current_path))
        {
            filelist->push_back(current_path.string());
        }
    }
    return true;
}

bool FindCorrespondingFiles(const std::vector<std::string>& filelist, const std::string& prefix, const std::string& extension, std::vector<std::string>* correspondingfilelist)
{
    if (!bfs::exists(prefix)) {
        printf("ERROR: %s does not exist.",prefix.c_str());
        return false;
    }
    for(int i = 0; i < filelist.size(); ++i)
    {
        bfs::path cur_path = (bfs::path(prefix)/(bfs::basename(filelist[i])+extension));
        if (bfs::exists(cur_path)) {
            correspondingfilelist->push_back(cur_path.string());
          } else {
            correspondingfilelist->push_back("");
          }
    }
    return true;
}

void ReadDepthMaps(const std::vector<std::string>& filelist,
                   std::vector<std::unique_ptr<cv::Mat>>& depthmaps)
{
    depthmaps.clear();
    for(int i=0; i < filelist.size(); ++i)
    {
        depthmaps.push_back(std::unique_ptr<cv::Mat>(new cv::Mat(cv::imread(filelist[i], -1))));
        //imread -1: Return the loaded image as is (with alpha channel).
    }
}

void ReadImages(const std::vector<std::string>& filelist,
                   std::vector<std::unique_ptr<cv::Mat>>& images)
{
    images.clear();
    for(int i=0; i < filelist.size(); ++i)
    {
        images.push_back(std::unique_ptr<cv::Mat>(new cv::Mat(cv::imread(filelist[i]))));
    }
}

bool ReadCameraInfos(const std::vector<std::string>& param_filelist,
                     std::vector<RectifiedCameraPair>& cam_infos)
{
    cam_infos.clear();
    for (int i = 0; i < param_filelist.size(); ++i)
    {
        const string& cur_param_file = param_filelist[i];
        //cout << cur_param_file << endl;
        std::ifstream cur_param_ifstream;
        cur_param_ifstream.open(cur_param_file);
        if (!cur_param_ifstream.is_open())
        {
            fprintf(stderr,"\nERROR: Could not open File %s",
                    cur_param_file.c_str());
            return false;
        }
        else
        {
            RectifiedCameraPair new_pair;
            cur_param_ifstream >> new_pair;
            cam_infos.push_back(new_pair);
        }
    }
    return true;
}

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 const std::string& semantic_label_prefix,
                                 std::vector<std::string>* cam_info_files,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files,
                                 std::vector<std::string>* confidence_files,
                                 std::vector<std::string>* semantic_files,
                                 bool use_confidence)
{
    using bfs::path;
    path root_folder(urban_data_root);
    if(!bfs::exists(root_folder)) {
        printf("ERROR: %s does not exist.\n",urban_data_root.c_str());
        return false;
    }
    path param_folder = root_folder/param_prefix;
    path image_folder = root_folder/image_prefix;
    path depth_folder = root_folder/depth_prefix;

    if(!ListFiles(depth_folder.string(), ".png", depth_files)) return false;
    if(!FindCorrespondingFiles(*depth_files, param_folder.string(), ".txt", cam_info_files)) return false;
    if(!FindCorrespondingFiles(*depth_files, image_folder.string(), ".png", image_files)) return false;

    if (use_confidence)
    {
        path confidence_folder = depth_folder.string() + "_confidence";
        if(!FindCorrespondingFiles(*depth_files, confidence_folder.string(), ".png", confidence_files)) return false;
    }

    if (semantic_label_prefix.empty() || !semantic_files) return true;
    path semantic_label_folder = root_folder/semantic_label_prefix;
    if (!bfs::exists(semantic_label_folder)) return false;
    if (!FindCorrespondingFiles(*depth_files, semantic_label_folder.string(), ".txt", semantic_files)) return false;
    return true;
}

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 const std::string& semantic_label_prefix,
                                 std::vector<RectifiedCameraPair>* cam_infos,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files,
                                 std::vector<std::string>* confidence_files,
                                 std::vector<std::string>* semantic_files, bool use_confidence)
{
    using bfs::path;
    std::vector<std::string> param_filelist;
    if(!LoadUrbanReconstructionData(urban_data_root,
                               param_prefix,
                               image_prefix,
                               depth_prefix,
                               semantic_label_prefix,
                               &param_filelist,
                               image_files,
                               depth_files,
                               confidence_files,
                               semantic_files, use_confidence))
        return false;
    if(!ReadCameraInfos(param_filelist, *cam_infos)) return false;
    return true;
}


bool LoadUrbanReconstructionData(
        const std::string &urban_data_root,
        const std::string &param_prefix,
        const std::string &image_prefix,
        const std::string &depth_prefix,
        const std::string &semantic_label_prefix,
        const int start_frame, const int end_frame,
        std::vector<std::string> *cam_info_files,
        std::vector<std::string> *image_files,
        std::vector<std::string> *depth_files,
        std::vector<std::string> *confidence_files,
        std::vector<std::string> *semantic_files,
        bool use_confidence)
{
    using bfs::path;
    path root_folder(urban_data_root);
    if(!bfs::exists(root_folder))
    {
        printf("root folder doesn't exist\n");
        return false;
    }
    path param_folder = root_folder/param_prefix;
    path image_folder = root_folder/image_prefix;
    path depth_folder = root_folder/depth_prefix;

    // if(!ListFiles(depth_folder.string(), ".png", depth_files)) return false;
    if (!ListFilesWithinFrameRange(depth_folder.string(), ".png", start_frame, end_frame, depth_files))
    {
        printf("depth maps not found.\n");
        return false;
    }
    if(!FindCorrespondingFiles(*depth_files, param_folder.string(), ".txt", cam_info_files))
    {
        printf("param files not found.\n");
        return false;
    }
    if(!FindCorrespondingFiles(*depth_files, image_folder.string(), ".png", image_files))
    {
        printf("image files not found.\n");
        return false;
    }

    if (use_confidence)
    {
        path confidence_folder = depth_folder.string() + "_confidence";
        if(!FindCorrespondingFiles(*depth_files, confidence_folder.string(), ".png", confidence_files))
        {
                printf("confidence files not found.\n");
                return false;
        }
    }

    if (semantic_label_prefix.empty() || !semantic_files) return true;
    path semantic_label_folder = root_folder/semantic_label_prefix;
    if (!bfs::exists(semantic_label_folder)) return false;
    if (!FindCorrespondingFiles(*depth_files, semantic_label_folder.string(), ".txt", semantic_files))
    {
        printf("semantic files not found.\n");
        return false;
    }
    return true;
}


bool LoadUrbanReconstructionData(
        const std::string &urban_data_root,
        const std::string &param_prefix,
        const std::string &image_prefix,
        const std::string &depth_prefix,
        const std::string &semantic_label_prefix,
        const int start_frame, const int end_frame,
        std::vector<RectifiedCameraPair> *cam_infos,
        std::vector<std::string> *image_files,
        std::vector<std::string> *depth_files,
        std::vector<std::string> *confidence_files,
        std::vector<std::string> *semantic_files,
        bool use_confidence)
{
    using bfs::path;
    std::vector<std::string> param_filelist;
    if(!LoadUrbanReconstructionData(urban_data_root,
                                    param_prefix,
                                    image_prefix,
                                    depth_prefix,
                                    semantic_label_prefix,
                                    start_frame,
                                    end_frame,
                                    &param_filelist,
                                    image_files,
                                    depth_files,
                                    confidence_files,
                                    semantic_files, use_confidence))
        return false;
    if(!ReadCameraInfos(param_filelist, *cam_infos)) return false;
    return true;
}



