/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <string>
#include <vector>
#include <memory>

class RectifiedCameraPair;
namespace cv {
class Mat;
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
                                 std::vector<std::string>* semantic_files,  bool use_confidence);

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 const std::string& semantic_label_prefix,
                                 std::vector<std::string>* cam_info_files,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files,
                                 std::vector<std::string>* confidence_files,
                                 std::vector<std::string>* semantic_files, bool use_confidence);

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 const std::string& semantic_label_prefix,
                                 const int start_frame,
                                 const int end_frame,
                                 std::vector<std::string>* cam_info_files,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files,
                                 std::vector<std::string>* confidence_files,
                                 std::vector<std::string>* semantic_files, bool use_confidence);

bool LoadUrbanReconstructionData(const std::string& urban_data_root,
                                 const std::string& param_prefix,
                                 const std::string& image_prefix,
                                 const std::string& depth_prefix,
                                 const std::string& semantic_label_prefix,
                                 const int start_frame,
                                 const int end_frame,
                                 std::vector<RectifiedCameraPair>* cam_infos,
                                 std::vector<std::string>* image_files,
                                 std::vector<std::string>* depth_files,
                                 std::vector<std::string>* confidence_files,
                                 std::vector<std::string>* semantic_files,  bool use_confidence);

cv::Mat LoadSemanticLabelAsMat(const std::string & semantic_label_filename);

bool ListFiles(const std::string& dir, const std::string& extension, std::vector<std::string>* filelist);

bool ListDirs(const std::string& root_dir, const std::string& prefix, std::vector<std::string>* dirlist);

bool FindCorrespondingFiles(const std::vector<std::string>& filelist, const std::string& prefix, const std::string& extension, std::vector<std::string>* correspondingfilelist);

bool ReadCameraInfos(const std::vector<std::string>& param_filelist,
                     std::vector<RectifiedCameraPair>& cam_infos);

void ReadDepthMaps(const std::vector<std::string>& filelist,
                   std::vector<std::unique_ptr<cv::Mat>>& depthmaps);

void ReadImages(const std::vector<std::string>& filelist,
                   std::vector<std::unique_ptr<cv::Mat>>& images);

bool ListFilesWithinFrameRange(
        const std::string& dir,
        const std::string& ext,
        int start_image,
        int end_image,
        std::vector<std::string>* filelist);

