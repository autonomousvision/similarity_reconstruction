#!/bin/bash

# directory which contains the image sequence
root_dir_left=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_02/rect
root_dir_right=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_03/rect
data_roots="$root_dir_left $root_dir_right"
cam_info_prefix="param_scale_4"
skymap_prefix="sky_00_scale_4"
depth_prefix="depth_00_slic_cropped_scale_4_filtered"

# root folder of binary files
#bin_dir=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/
bin_dir=/home/dell/codebase/mpi_project/urban_reconstruction/code/build_handin/bin/
bin_dir=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction-build/bin/

# folder containing third party software
third_party_dir=/home/dell/link_to_urban_recon/third_party/

result_root=/home/dell/results_5/

detector_train_data_dir=/home/dell/results_5/detection_training_data/

