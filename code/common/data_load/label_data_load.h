#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


typedef pcl::PointXYZ PointXYZ;

class Cube
{
 public:
      std::vector<PointXYZ> vertices;
      int * edges;
      int * faces;
      PointXYZ center;
  
      std::string name;
      std::string category;
      float level_min, level_max;
      float min_x, min_y, min_z, max_x, max_y, max_z;
  
      Eigen::Matrix4f transform;
  
      int fNum, vNum, eNum;
  
      Cube() {
          center.x = 0;
          center.y = 0;
          center.z = 0;

          edges = NULL;
          faces = NULL;
      }
  
      ~Cube() {
          if (edges != NULL) delete edges;
          if (faces != NULL) delete faces;
      }

      void computeEdges(const cv::Mat &faces);

      void computeFaces(const cv::Mat &faces);
  
      void initGeometry(int v, int f);
  
      void transformCube(const Eigen::Matrix4f &trans);
 
      void computeBoundingbox();
 };


/* Init Cube geometry */


/* Compute Faces */


/* Compute Edges */



/* Transform the cube to the annotated pose */


/* Compute the smalled bounding box for the cube */



/* Parse a single xml obj */
bool readSingleAnnotation(cv::FileNode &item, Cube *cube);


/*
Parse xml file
@params: 
	fileName: name of the xml
	objects_cube: vector of the bounding boxes in xml file (return)
@return:
	number of bounding boxes
*/
 int readUserDataXML(const char *fileName, std::vector<Cube*> &objects_cube);

 int getLabeledCubes(const std::vector<Cube*> & objects_cube, const std::string& name, std::vector<Cube*>* cate_object_cubes);

 int convertCubeToAffine(const std::vector<Cube*>& cubes, std::vector<Eigen::Affine3f>* transforms);

 void freeCubes(std::vector<Cube*> &objects_cube);

 //int readUserDataXML(char * fileName, const std::string& name, std::vector<Cube*> &objects_cube)

