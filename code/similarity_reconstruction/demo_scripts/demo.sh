#!/bin/bash

. ./init_paths.sh
run_train=0
run_detect=1
run_joint_opt=1
. ./run_detection.sh
. ./run_joint_reconstruction.sh
