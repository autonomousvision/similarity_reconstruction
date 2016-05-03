/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#pragma once
#include <vector>
#include <string>
namespace cv
{
class Mat;
}
class RectifiedCameraPair;

template<typename T>
inline void writeFalseColors(const cv::Mat_<T>& original_image, const std::string& filename, float max_val = -1)
{
    assert(original_image.channels() == 1);

    // color map
    float map[8][4] = {{0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
        {0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0}
    };
    float sum = 0;
    for (int32_t i=0; i<8; i++)
        sum += map[i][3];

    float weights[8]; // relative weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int32_t i=0; i<7; i++)
    {
        weights[i]  = sum/map[i][3];
        cumsum[i+1] = cumsum[i] + map[i][3]/sum;
    }

    // create color png image
    cv::Mat image(original_image.rows, original_image.cols, CV_8UC3);
    int height = original_image.rows;
    int width = original_image.cols;

    if (max_val == -1)
    {
        max_val = std::numeric_limits<T>::max();
    }

    // for all pixels do
    for (int32_t v=0; v<height; v++)
    {
        for (int32_t u=0; u<width; u++)
        {

            // get normalized value
            float val = std::min<double>(std::max<double>(double(original_image(v, u))/max_val, 0.0f),1.0f);

            // find bin
            int32_t i;
            for (i=0; i<7; i++)
                if (val<cumsum[i+1])
                    break;

            // compute red/green/blue values
            float   w = 1.0-(val-cumsum[i])*weights[i];
            uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
            uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
            uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

            // set pixel
            //image.set_pixel(u,v,png::rgb_pixel(r,g,b));
            image.at<cv::Vec3b>(v, u) = cv::Vec3b(b,g,r);
        }
    }

    // write to file
    cv::imwrite(filename, image);
}


void DebugPointCloud(const std::vector<RectifiedCameraPair>& cam_infos,
                     const std::vector<std::unique_ptr<cv::Mat>>& depths,
                     const std::vector<std::unique_ptr<cv::Mat>>& images,
                     double maxCamDistance,
                     std::vector<cv::Vec3d>& points3d, std::vector<cv::Vec3i>& pointscolor);

void WritePointCloudToFile(const std::string& filename, const std::vector<cv::Vec3d>& point3d, const std::vector<cv::Vec3i>& pointscolor);

void Fusion_SphericalError(const std::vector<RectifiedCameraPair>& cam_infos,
                  const std::vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const std::vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir = std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug"));


void VisibilityFusion(const std::vector<RectifiedCameraPair>& cam_infos,
                      const std::vector<std::unique_ptr<cv::Mat>>& depths,
                      cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance);

void SimpleFusion(const std::vector<RectifiedCameraPair>& cam_infos,
                  const std::vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const std::vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  const std::string& save_dir = std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")
                  );

void SimpleFusionDisparity(const std::vector<RectifiedCameraPair>& cam_infos,
                  const std::vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const std::vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir = std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")
                  );

void SimpleFusionDisparity_onlyrefsupport(const std::vector<RectifiedCameraPair>& cam_infos,
                  const std::vector<std::unique_ptr<cv::Mat>>& depths,
                  cv::Mat* fusioned_map, cv::Mat* confidence_map, double maxCamDistance,
                  const std::vector<std::string>& param_file_list,
                  const std::vector<std::string>& image_file_list,
                  double support_thresh,
                  const std::string& save_dir = std::string("/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/debug")
                  );
