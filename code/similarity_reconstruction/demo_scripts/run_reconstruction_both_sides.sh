#!/bin/bash
# stops when error
set -e
# run reconstruction pipline (using the one computing true TSDF) for both sides of the street
# set relevant paths 
. ./init_paths.sh

#startimg=1470
#endimg=1790
#startimg=1550
#endimg=1630
echo "Performing reconstruction from frame "$startimg" to "$endimg" for both sides."
sleep 1

if [ $run_reconstruction -gt 0 ]; then
    do_depth2ply=1
    do_ply2vri=1
    do_fusevri=1
else 
    do_depth2ply=0
    do_ply2vri=0
    do_fusevri=0
fi

echo "side 1"
root_dir=$root_dir_left
# the suffix to differentiate different runs
run_suffix="_side1"
use_input_tsdf_file=0
. ./run_reconstruction_one_side.sh

echo "side 2"
root_dir=$root_dir_right
run_suffix="_side2_with_side1"
use_input_tsdf_file=1
input_tsdf_file_path=$output_tsdf_file
. ./run_reconstruction_one_side.sh

echo ./visualization.sh "initial_reconstruction" "$output_tsdf_ply_file"
. ./visualization.sh "initial_reconstruction" "$output_tsdf_ply_file"  
scene_model_mesh=$output_tsdf_ply_file
scene_model_bin=$output_tsdf_file

