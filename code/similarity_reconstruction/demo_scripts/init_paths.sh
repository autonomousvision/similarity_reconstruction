#!/bin/bash

# directory which contains the image sequence
# this group of variables are only needed if we need to run the reconstruction from image sequences
data_root_dir=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/
root_dir_left=$data_root_dir/image_02/rect/
root_dir_right=$data_root_dir/image_03/rect/
data_roots="$root_dir_left $root_dir_right"
# the prefix of pre-generated files: camera parameters, depth maps, sky labels
# should be placed under rect/ folders
cam_info_prefix=param_scale_4
image_prefix=img_00_scale_4
depth_prefix=depth_00_slic_cropped_scale_4_filtered
skymap_prefix=sky_00_scale_4
max_cam_distance=30
startimg=1470
endimg=1790

# root folder of binary files
bin_dir=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction-build/bin/

# folder containing third party softwares
# third_party_dir=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/third_party/

# folder to store results
result_root=/home/dell/results_5/recon_demo_$startimg"_"$end/
if [ ! -d $result_root ]; then
    mkdir $result_root
fi

# the binary file for visualization
mesh_view_bin=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction/visualization/trimesh2/bin.Linux64/mesh_view
display=1

# root for demo data
demo_data_root=/home/dell/results_5/training_data/
# used for object detection training part: folder storing training data for object detection
detector_train_data_dir=$demo_data_root/training_scene/

# the path for pretrained detectors and initial 3D reconstruction
detector_file_dir=$demo_data_root/detectors/
scene_model_bin=$demo_data_root/reconstructing_scene/recon_tsdf.bin

