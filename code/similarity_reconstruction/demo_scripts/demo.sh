#!/bin/bash

. ./init_paths.sh
run_train=0
run_detect=0
run_joint_opt=1
detector_train_data_dir=/home/dell/results_5/demo_1470_1790-0.4bthresh/cropped_tsdf_for_training/train_data/
detector_file_dir=/home/dell/results_5/demo_1470_1790-0.4bthresh/run_detection/
scene_model_bin=/home/dell/results_5/demo_1470_1790-0.4bthresh/reconstruction_baseline_1470_1790/vri_fusing_result_side2_with_side1_1470_1790/recon-1470-1790_tsdf.bin
. ./run_detection.sh
. ./run_joint_reconstruction.sh
