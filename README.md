# Similarity Reconstruction
Objects of similar types and shapes are common in urban areas. In this project we take advantage of this observation to improve the quality of 3D reconstruction. Similarly shaped objects are located using detectors and then jointly reconstructed by learning a volumetric model. By aggregating observations from multiple instances, the completeness of the reconstruction can be improved and noise can be suppressed.

### Related Paper
* **Exploiting Object Similarity in 3D Reconstruction**, *Chen Zhou, Fatma Güney, Yizhou Wang , Andreas Geiger*, ICCV '15

# 1. Prerequisites
The code is written and tested under Ubuntu 14.04.3. The following libraries are required:

  1.  Boost (>=1.54)
  2.  PCL (>=1.7)
  3.  OpenCV (>=2.4.6)
  4.  Eigen (>=3.2.4)
  5.  [Ceres solver](http://ceres-solver.org/) (>=1.10)
  6.  [Glog](https://github.com/google/glog)

Boost, OpenCV and Glog can be installed from the repository:

```
sudo apt-get install libboost-all-dev libopencv-dev libgoogle-glog-dev 
```
The other libraries needs to be installed manually:

  - Eigen: The eigen package in the ubuntu 14.04 repository is 3.2.0. It's recommended to download a newer version of Eigen from its [website](http://eigen.tuxfamily.org/index.php?title=Main_Page).

  - Ceres-solver: The library and instructions for installing can be donwloaded from [source](http://ceres-solver.org/).

  - [Point Cloud Library](http://pointclouds.org/downloads/linux.html): 
  ```
  sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
  sudo apt-get update
  sudo apt-get install libpcl-all
  ```

# 2. Downloads
Download the project:

```sh
git clone http://cvlibs.net:3000/ageiger/similarity_reconstruction.git similarity_reconstruction
```

Create a data folder and extract the following zip file into this folder:

[Sequence files & trained models (294 MB)](http://www.cvlibs.net/download.php?file=similarity_reconstruction_data.zip): 

# 3. Compiling the Code

Compile the code:

```sh
cd similarity_reconstruction/code/similarity_reconstruction
mkdir build
cd build
cmake ..
make -j 6
```

Compile the mesh viewer for visualizing the results (optional).
In the root folder of the code:

```
cd similarity_reconstruction/code/similarity_reconstruction/visualization/trimesh2
make
```
Note that the output ply files can be directly viewed in Meshlab, Cloudcompare, etc.

# 4. Running the Code

Several bash scripts in `similarity_reconstruction/code/similarity_reconstruction/demo_scripts/` are used to run the code.

To run the demos one needs to download the pre-processed image sequence data, the initial reconstruction and the training files (see link above).

* Running `demo.sh`: the demo script runs object detection and joint reconstruction using pretrained 3D detectors and the initial 3D reconstruction.

    1. Set the follwing variables in `init_paths_samples.sh`, rename it to `init_paths.sh` and make it executable (chmod a+x init_paths.sh):

        -`$code_dir`: the root folder of the code (where you have cloned the git repository)

        -`$data_dir`: the folder where you have extracted the data zip file

    2. Run `demo.sh`.

    The results are stored in data_dir/results. After completion of the program, a 3D viewer should appear which displays the results in 3D.

* Running `run_all.sh`: the scripts runs the whole pipeline, including initial 3D reconstruction from image sequence, detector training, detection and joint optimization.

    1. Set the follwing variables in `init_paths.sh`:

        -`$code_dir`: the root folder of the code (where you have cloned the git repository)

        -`$data_dir`: the folder where you have extracted the data zip file

        - Multiple `*prefix` variables: specify paths for pre-calculated camera parameters, image sequences, depth maps and sky segmentations. The current system uses down-scaled versions of the image sequence, semi-global matching based and consistency filtered depth maps as well as segmentation masks of the sky regions. For the downloaded demo sequence these paths can be left at their pre-specified values as these modalities have been pre-computed. For new sequences, this data has to be generated. However, as this code has a lot of dependencies and is not refactored so far it is not included.

    2. Run `run_all.sh`.

![screenshot](screenshot.png "screenshot")

# 5. Comments & Questions

If you have comments, questions or suggestions, please contact the first author of the paper, [Chen Zhou](mailto:zhouch@pku.edu.cn).

