#include "label_data_load.h"
#include <set>


void Cube::computeEdges(const cv::Mat &faces)
{
    using namespace std;
    if ((category == "instance" && name != "tree") || category == "stuff") {
        eNum = vNum*3/2;
        edges = (int *) malloc(eNum*2*sizeof(int));

        // cube edges
        if (category == "instance") {
            for (int i = 0; i < 2; i++) {
                edges[8*i] = 2*i;      edges[8*i+1] = 2*i+1;
                edges[8*i+2] = 2*i+1;  edges[8*i+3] = 2*i+4;
                edges[8*i+4] = 2*i+4;  edges[8*i+5] = 2*i+5;
                edges[8*i+6] = 2*i+5;  edges[8*i+7] = 2*i;
            }

            edges[16] = 0; edges[17] = 2;
            edges[18] = 1; edges[19] = 3;
            edges[20] = 4; edges[21] = 6;
            edges[22] = 5; edges[23] = 7;
        }

        // polygon edges
        else if (category == "stuff"){
            int v = vertices.size()/2;
            for (int i = 0; i < v; i++) {
                if (i < v-1) {
                    edges[i*2] = i; edges[i*2+1] = i+1;
                    edges[i*2 + v*2] = i+v; edges[i*2+1 + v*2] = i+1+v;
                }
                edges[i*2 + v*4] = i; edges[i*2 + v*4 + 1] = i+v;
            }
            edges[2*v-2] = v-1; edges[2*v-1] = 0;
            edges[4*v-2] = 2*v-1; edges[4*v-1] = v;
        }
    }

    // tree primitives or stuff
    else {
        set< pair<int, int> > edgeSet;
        for (int i = 0; i < faces.rows; i++) {
            int maxV = max(faces.at<uchar>(i, 0), faces.at<uchar>(i, 1));
            int minV = min(faces.at<uchar>(i, 0), faces.at<uchar>(i, 1));
            edgeSet.insert(make_pair(minV, maxV));

            maxV = max(faces.at<uchar>(i, 1), faces.at<uchar>(i, 2));
            minV = min(faces.at<uchar>(i, 1), faces.at<uchar>(i, 2));
            edgeSet.insert(make_pair(minV, maxV));

            maxV = max(faces.at<uchar>(i, 0), faces.at<uchar>(i, 2));
            minV = min(faces.at<uchar>(i, 0), faces.at<uchar>(i, 2));
            edgeSet.insert(make_pair(minV, maxV));
        }

        eNum = edgeSet.size();
        edges = (int *) malloc(eNum*2*sizeof(int));

        set< pair<int, int> >::iterator it;
        int count = 0;
        for(it = edgeSet.begin(); it != edgeSet.end(); ++it) {
            edges[count] = (*it).first;
            edges[count+1] = (*it).second;
            count += 2;
        }
    }

}

void Cube::transformCube(const Eigen::Matrix4f &trans)
{
    for (int i =0 ;i <vNum; i++) {
        vertices[i] = pcl::transformPoint (vertices[i], Eigen::Affine3f(trans));
    }

    center = pcl::transformPoint (center, Eigen::Affine3f(trans));
}

void Cube::computeFaces(const cv::Mat &faces)
{
    for (int i = 0; i < faces.rows; i++) {
        this->faces[i*3] = faces.at<uchar>(i, 0);
        this->faces[i*3+1] = faces.at<uchar>(i, 1);
        this->faces[i*3+2] = faces.at<uchar>(i, 2);
    }
}


void Cube::initGeometry(int v, int f)
{
    fNum = f;
    vNum = v;
    faces = (int *) malloc(f*3*sizeof(int));
}


void Cube::computeBoundingbox()
{
    min_x = 100000, max_x = -100000, min_y = 100000, max_y = -100000, min_z = 100000, max_z = -100000;
    for (size_t i = 0; i < vertices.size(); i++)
    {
        if (vertices[i].x > max_x) {
            max_x = vertices[i].x;
        }
        if (vertices[i].y > max_y) {
            max_y = vertices[i].y;
        }
        if (vertices[i].z > max_z) {
            max_z = vertices[i].z;
        }

        if (vertices[i].x < min_x) {
            min_x = vertices[i].x;
        }
        if (vertices[i].y < min_y) {
            min_y = vertices[i].y;
        }
        if (vertices[i].z < min_z) {
            min_z = vertices[i].z;
        }
    }
}


bool readSingleAnnotation(cv::FileNode &item, Cube *cube)
{
    using namespace std;
    try {
    using namespace std;
    cout << "read object" << endl;
    int index;
    item["index"] >> index;
    cout << "index: " << index << endl;
    item["label"] >> cube->name;
    item["category"] >> cube->category;
    item["level_min"] >> cube->level_min;
    item["level_max"] >> cube->level_max;
    cout << "label: " << cube->name << endl;
    cout << "category: " << cube->category << endl;
    cout << "level_min: " << cube->level_min << endl;
    cout << "level_max: " << cube->level_max << endl;

    cv::Mat transform;
    item["transform"] >> transform;
    // convert from cvMat to eigen::matrix4f
    for (int i = 0; i < 16; i++) {
        int xi = i/4;
        int yi = i%4;
        cube->transform(xi, yi) = transform.at<float>(xi, yi);
    }
    cout << "transform: \n" << transform << endl;

    cv::Mat vertices, faces;
    item["vertices"] >> vertices;
    item["faces"] >> faces;

    cout << "read obj finished " << index<< endl;

    cube->initGeometry(vertices.rows, faces.rows);

    cube->center = PointXYZ(0, 0, 0);
    for (int i = 0; i < vertices.rows; i++) {
        cube->vertices.push_back(PointXYZ(vertices.at<float>(i, 0), vertices.at<float>(i, 1),
                                          vertices.at<float>(i, 2)));
        cube->center.x += vertices.at<float>(i, 0);
        cube->center.y += vertices.at<float>(i, 1);
        cube->center.z += vertices.at<float>(i, 2);
    }

    cube->center.x /= vertices.rows;
    cube->center.y /= vertices.rows;
    cube->center.z /= vertices.rows;

    cube->computeFaces(faces);
    cube->computeEdges(faces);
    }
    catch (cv::Exception& e)
    {
        cout << "cv exception " << endl;
        cout << e.what() << endl;
        return false;
    }
    return true;
}


int readUserDataXML(const char *fileName, std::vector<Cube *> &objects_cube)
{
    using namespace std;
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        return -1;
    }


    cv::FileNode n = fs.root();
    int num = 0;

    // parse version
    int version = -1;
    fs["version"] >> version;
    cout << "xml version: " << version << endl;

    cv::FileNodeIterator current = n.begin();

    string userName;
    string taskId;
    if (version == 1) {
        current ++;
        fs["taskID"] >> taskId;
        cout << taskId << endl;
        current ++ ;
        fs["userId"] >> userName;
        cout << userName << endl;
        current ++;
    }


    std::vector<int> fail_num;
    for ( ; current != n.end(); current++) {
        cv::FileNode item = *current;
        // read a single object
        Cube *cube = new Cube();
        if (readSingleAnnotation(item, cube))
        {
            objects_cube.push_back(cube);
        }
        else
        {
            fail_num.push_back(num + 1);
            delete cube;
        }
        num ++;
    }
    cout << "Read object total: " << objects_cube.size() << endl;
    cout << "fail objects:" << endl;
    for (int i = 0; i < fail_num.size(); ++i)
    {
        cout << "fail obj: " << fail_num[i] << endl;
    }

    for (size_t i = 0; i < objects_cube.size(); i++) {
        //transform to the world cs
        objects_cube[i]->transformCube(objects_cube[i]->transform);
        objects_cube[i]->computeBoundingbox();
    }

    return num;
}


void freeCubes(std::vector<Cube *> &objects_cube)
{
    for (int i = 0; i < objects_cube.size(); ++i)
    {
        delete objects_cube[i];
    }
}


int convertCubeToAffine(const std::vector<Cube *> &cubes, std::vector<Eigen::Affine3f> *transforms)
{
    for (int i = 0; i < cubes.size(); ++i)
    {
        transforms->push_back(Eigen::Affine3f(cubes[i]->transform));
    }
    return 1;
}


int getLabeledCubes(const std::vector<Cube *> &objects_cube, const std::string &name, std::vector<Cube *> *cate_object_cubes)
{
    for (int i  = 0; i < objects_cube.size(); ++i)
    {
        if (objects_cube[i]->name == name)
        {
            cate_object_cubes->push_back(objects_cube[i]);
        }
    }
    return 1;
}
