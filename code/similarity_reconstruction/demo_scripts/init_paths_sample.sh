#!/bin/bash

# code directory and data directory
code_dir=/home/ageiger/4_Projects/cvlibs_git/similarity_reconstruction
data_dir=/media/ageiger/data/projects/similarity_reconstruction/release

# directory which contains the image sequence
data_root_dir=$data_dir/sequences/2013_05_28/2013_05_28_drive_0000_sync/
root_dir_left=$data_root_dir/image_02/rect/
root_dir_right=$data_root_dir/image_03/rect/
data_roots="$root_dir_left $root_dir_right"

# prefix of pre-generated files: camera parameters, depth maps, sky labels (in rect/ subfolders)
cam_info_prefix=param_scale_4
image_prefix=img_00_scale_4
depth_prefix=depth_00_slic_cropped_scale_4_filtered
skymap_prefix=sky_00_scale_4
max_cam_distance=30
startimg=1470
endimg=1790

# root folder of binary files
bin_dir=$code_dir/code/similarity_reconstruction/build/bin/

# third party software
third_party_dir=$code_dir/code/third_party/

# folder to store results
result_root=$data_dir/results/recon_demo_$startimg"_"$endimg/
if [ ! -d $result_root ]; then
  mkdir $result_root
fi

# the binary file for visualization
mesh_view_bin=$code_dir/code/similarity_reconstruction/visualization/trimesh2/bin.Linux64/mesh_view
display=1

# training data
demo_data_root=$data_dir/training_data/
detector_train_data_dir=$demo_data_root/training_scene/

# pretrained detectors and initial 3D reconstruction
detector_file_dir=$demo_data_root/detectors/
scene_model_bin=$demo_data_root/reconstructing_scene/recon_tsdf.bin

