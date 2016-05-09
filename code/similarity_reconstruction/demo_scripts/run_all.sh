#!/bin/bash

run_reconstruction=1
run_train=1
run_detect=1
run_joint_opt=1
. ./init_paths.sh
. ./run_reconstruction_both_sides.sh
. ./run_detection.sh
. ./run_joint_reconstruction.sh
