/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <memory>

//#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include "tclap/CmdLine.h"
#include "common/fisheye_camera/RectifiedCameraPair.h"
#include "common/data_load/urban_reconstruction_data_load.h"
#include "depthmap_noise_filter/depthmap_filter.h"

namespace bfs = boost::filesystem;
using namespace std;

inline bool FileExists(const std::string& filepath)
{
	return bfs::exists(filepath);
}

bool AppendFilesOfOneFrameBoth(const string& param_dir,
                               int cur_frame,
                               std::vector<std::string>& param_filelist, int neighbor_limit = 1)
{
    char cur_frame_name[256];
    // add reference frame
    int ref_parired_frame = cur_frame + 1;
    sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, ref_parired_frame);
    if (!FileExists(param_dir + cur_frame_name))
    {
        ref_parired_frame = cur_frame - 1;
        sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, ref_parired_frame);
        if(!FileExists(param_dir + cur_frame_name)) return false;  // no reference pair
    }
    param_filelist.push_back(param_dir + cur_frame_name);
    // add ref_neighboring frames
    for (int i = cur_frame - neighbor_limit; i <= cur_frame + neighbor_limit; ++i)
    {
        if (ref_parired_frame == i) continue; // reference pair, already added.
        sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, i);
        if (FileExists(param_dir + cur_frame_name))
        {
            param_filelist.push_back(param_dir + cur_frame_name);
        }
    }
    return true;
}

bool AppendFilesOfOneFrame(const string& param_dir,
                           int cur_frame,
                           std::vector<std::string>& param_filelist, int neighbor_limit, bool useright)
{
    char cur_frame_name[256];
    int sign = useright? 1:-1;
    // add reference frame
    int ref_parired_frame = cur_frame + 1*sign;
    sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, ref_parired_frame);
    if (!FileExists(param_dir + cur_frame_name))
    {
        return false;
    }
    param_filelist.push_back(param_dir + cur_frame_name);
    // add ref_neighboring frames
    for (int i = cur_frame - neighbor_limit; i <= cur_frame + neighbor_limit; ++i)
    {
        if (ref_parired_frame == i) continue; // reference pair, already added.
        sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, i);
        if (FileExists(param_dir + cur_frame_name))
        {
            param_filelist.push_back(param_dir + cur_frame_name);
        }
    }
    return true;
}

bool ConstructFileLists(const string& root_dir,
                        int cur_frame,
                        std::vector<std::string>& param_filelist, bool useright)
{
    param_filelist.clear();
    string param_dir = root_dir + "/param2/";
    // add reference pair and pairs with cur_frame as ref. frame
    if (!AppendFilesOfOneFrame(param_dir, cur_frame, param_filelist, 0, useright)) return false;
    // add other neigoboring frames
    const int other_ref_frame_limit = 2;
    for (int i = cur_frame - other_ref_frame_limit; i <= cur_frame + other_ref_frame_limit; ++i)
    {
        if (i == cur_frame) continue;
        //AppendFilesOfOneFrameBoth(param_dir, i, param_filelist, 1);
        AppendFilesOfOneFrameBoth(param_dir, i, param_filelist, 1);
    }
    return true;
}

// use_right = 1: use <current_frame, next_paired_frame> as the reference depth map, if it doesn't exist, return false.
// use_right = 2: use <current_frame, previous_paired_frame> as the reference depth map, if it doesn't exist, return false.
// use_both = 3: append both sides to param_filelist.
bool AppendFilesOfOneFrame2(const string& param_dir, int cur_frame, std::vector<std::string>& param_filelist, int use_right)
{
    bool res = false;
    char cur_frame_name[256];
    int i = 1;
    const int max_search_range = 20;
    if (use_right & 0x1)
    {
        while (i < max_search_range)
        {
            int paired_frame = cur_frame + i;
            sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, paired_frame);
            if (bfs::exists(bfs::path(param_dir) / bfs::path(cur_frame_name)))
            {
                param_filelist.push_back((bfs::path(param_dir) / bfs::path(cur_frame_name)).string());
                res = true;
                break;
            }
            ++i;
        }
    }
    if ((use_right>>1) & 0x1)
    {
        i = 1;
        while (i < max_search_range)
        {
            int paired_frame = cur_frame - i;
            sprintf(cur_frame_name, "%010d_%010d.txt", cur_frame, paired_frame);
            if (bfs::exists(bfs::path(param_dir) / bfs::path(cur_frame_name)))
            {
                param_filelist.push_back((bfs::path(param_dir) / bfs::path(cur_frame_name)).string());
                res = true;
                break;
            }
            ++i;
        }
    }
    return res;
}

bool ConstructFileLists2(const string& param_dir, int cur_frame, std::vector<std::string>& param_filelist, int use_right)
{
    param_filelist.clear();
    // add reference pair and pairs with cur_frame as reference frame
    if (!AppendFilesOfOneFrame2(param_dir, cur_frame, param_filelist, use_right)) return false;
    // add other neigoboring frames
    const int other_ref_frame_limit = 2;
    for (int i = cur_frame - other_ref_frame_limit; i <= cur_frame + other_ref_frame_limit; ++i)
    {
        if (i == cur_frame) continue;
        AppendFilesOfOneFrame2(param_dir, i, param_filelist, 3);
    }
    return true;
}

std::string FindCorrespondingOtherFile(const std::string& param_file, const std::string& depth_dir)
{
    using namespace boost::filesystem;
    path basename = path(param_file).stem();
    return std::string(depth_dir + "/" + basename.string() + ".png");
}

void FindCorrespondingOtherFiles(const std::vector<std::string>& param_files, const std::string& depth_dir, std::vector<std::string>& depth_files)
{
    depth_files.clear();
    for (int i=0; i< param_files.size(); ++i)
    {
        depth_files.push_back(FindCorrespondingOtherFile(param_files[i], depth_dir));
    }
}

int main(int argc, char *argv[])
{
    using namespace std;
    int startimage = 801;
    int endimage = 810;
    double maxCamDistance = 35.0;
    double depth_image_scaling_factor = maxCamDistance/65535;
    double support_thresh = 0.1;
    std::string output_prefix("depth-fusioned-cur");
    std::string depth_prefix("/depth_00_keep_saturate/");
    std::string rgb_prefix("/img_00/");
    std::string param_prefix("param2/");

    TCLAP::CmdLine cmdLine("onlinefusion");
    TCLAP::ValueArg<int> startimageArg("s","startimage",
                                       "Number of the Start Image",false,startimage,"int");
    TCLAP::ValueArg<int> endimageArg("e","endimage","Number of the End Image",false,endimage,"int");
    TCLAP::ValueArg<std::string> depthprefixArg("d","depthprefix", "Depth Prefix", true, depth_prefix, "string");
    TCLAP::ValueArg<std::string> rgbprefixArg("r","rgbprefix", "Rgb Prefix", true, rgb_prefix, "string");
    TCLAP::ValueArg<std::string> outputprefixArg("o","outputprefix", "Output Prefix", true, output_prefix, "string");
    TCLAP::ValueArg<double> maxCamDistanceArg("","max-camera-distance","Maximum Camera Distance to Surface",false,maxCamDistance,"double");
    TCLAP::ValueArg<double> supportThreshArg("","support-thresh","support-thresh",false, support_thresh, "double");
    TCLAP::ValueArg<std::string> paramprefixArg("", "paramprefix", "Param Prefix", false, param_prefix, "string");
    TCLAP::UnlabeledValueArg<std::string> associationfilenamesArg("rootdir", "The root dir",
            true, std::string(), "string");

    cmdLine.add(startimageArg);
    cmdLine.add(endimageArg);
    cmdLine.add(depthprefixArg);
    cmdLine.add(rgbprefixArg);
    cmdLine.add(outputprefixArg);
    cmdLine.add(associationfilenamesArg);
    cmdLine.add(maxCamDistanceArg);
    cmdLine.add(supportThreshArg);
    cmdLine.add(paramprefixArg);
    cmdLine.parse(argc,argv);

    startimage = startimageArg.getValue();
    endimage = endimageArg.getValue();
    depth_prefix = depthprefixArg.getValue();
    rgb_prefix = rgbprefixArg.getValue();
    output_prefix = outputprefixArg.getValue();
    param_prefix = paramprefixArg.getValue();
    maxCamDistance = maxCamDistanceArg.getValue();
    support_thresh = supportThreshArg.getValue();
    depth_image_scaling_factor = maxCamDistance/65535;
    string root_dir = associationfilenamesArg.getValue();
    string depth_dir = root_dir + "/" + depth_prefix + "/";
    string rgb_dir = root_dir + "/" + rgb_prefix + "/";
    string depth_output_dir = root_dir + "/" + output_prefix + "/";
    string conf_output_dir = root_dir + "/" + output_prefix + "_confidence" + "/";
    if (!bfs::exists( depth_output_dir ))
    {
        bfs::create_directory(depth_output_dir);
    }
    if (!bfs::exists( conf_output_dir ))
    {
        bfs::create_directory(conf_output_dir);
    }
    cout << "support threshold: " << support_thresh << endl;
    cout << "depth_output_dir: " << depth_output_dir << endl;
    cout << "conf_output_dir: " << conf_output_dir << endl;

    for (int i = startimage; i <= endimage; ++i)
    {
        cout << "current frame: " << i << endl;
        for (int useright = 1; useright <= 2; useright++)
        {
            cout << root_dir << endl;
            cout << param_prefix << endl;
            cout << (bfs::path(root_dir)/bfs::path(param_prefix)).string() << endl;
            vector<string> param_filelist;
            vector<string> depth_filelist;
            vector<string> image_filelist;
            if(!ConstructFileLists2((bfs::path(root_dir)/bfs::path(param_prefix)).string(), i, param_filelist, useright)) continue;
            for (int di = 0; di < param_filelist.size(); ++di)
            {
                std::cout << "Params: " << param_filelist[di]<<std::endl;
            }
            FindCorrespondingOtherFiles(param_filelist, depth_dir, depth_filelist);
            for (int di = 0; di < depth_filelist.size(); ++di)
            {
                std::cout << "Depths: " << depth_filelist[di]<<std::endl;
            }
            FindCorrespondingOtherFiles(param_filelist, rgb_dir, image_filelist);
            for (int di = 0; di < depth_filelist.size(); ++di)
            {
                std::cout << "Images: " << image_filelist[di]<<std::endl;
            }

            vector<RectifiedCameraPair> cam_infos;
            ReadCameraInfos(param_filelist, cam_infos);
            std::vector<std::unique_ptr<cv::Mat>> depth_maps;
            ReadDepthMaps(depth_filelist, depth_maps);
            for (int j = 0; j < cam_infos.size(); ++j)
            {
                cam_infos[j].SetVoxelScalingParameters(cv::Vec3d(0,0,0),
                                                       1.0,
                                                       depth_image_scaling_factor, maxCamDistance);
            }
            for (int di = 0; di < cam_infos.size(); ++di)
            {
                std::cout<<cam_infos[di]<<std::endl;
            }
            std::vector<std::unique_ptr<cv::Mat>> images;
            ReadImages(image_filelist, images);
            assert(cam_infos.size() == depth_maps.size());
            std::cout<< "Reading for " << i << " finished..." << std::endl;
            cv::Mat fusioned_map;
            cv::Mat confidence_map;
            //VisibilityFusion(cam_infos, depth_maps, &fusioned_map, &confidence_map, maxCamDistance);
            //SimpleFusion(cam_infos, depth_maps, &fusioned_map, &confidence_map, maxCamDistance, param_filelist, image_filelist);
            vector<cv::Vec3d> points3d;
            vector<cv::Vec3i> pointscolor;
            assert(images.size() == cam_infos.size());

           // SimpleFusionDisparity_onlyrefsupport(cam_infos, depth_maps, &fusioned_map, &confidence_map, maxCamDistance, param_filelist, image_filelist, support_thresh);
            Fusion_SphericalError(cam_infos, depth_maps, &fusioned_map, &confidence_map, maxCamDistance, param_filelist, image_filelist, support_thresh);
            std::cout << "visibility fusion for " << i << " finished..." << std::endl;
            {
            cv::Mat_<unsigned short> depth_map_16b;
            fusioned_map.convertTo(depth_map_16b, CV_16UC1, 1.0/depth_image_scaling_factor);
            cv::Mat_<unsigned short> conf_map_16b;
            confidence_map.convertTo(conf_map_16b, CV_16UC1, 1.0*65535);
            using namespace boost::filesystem;
            std::string depth_filename = depth_output_dir + "vis-" + path(param_filelist[0]).stem().string() + ".png";
            std::string conf_filename = conf_output_dir + "vis-" + path(param_filelist[0]).stem().string() + ".png";
            writeFalseColors(depth_map_16b, depth_filename);
            writeFalseColors(conf_map_16b, conf_filename);

            depth_filename = depth_output_dir + path(param_filelist[0]).stem().string() + ".png";
            conf_filename = conf_output_dir + path(param_filelist[0]).stem().string() + ".png";
            cout << depth_filename << endl;
            imwrite(depth_filename, depth_map_16b);
            imwrite(conf_filename, conf_map_16b);
            }
        }
    }

    return 0;
}



