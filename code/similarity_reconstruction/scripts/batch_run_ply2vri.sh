#!/bin/bash
set -e
# convert ply files in a directory to vri TSDF representation using ply2vri
input_dir=/home/dell/link_to_results/output-depthmap-meshing-world/
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
output_prefix=$output_root/tovri/
if [ ! -d "$output_prefix" ]; then
mkdir "$output_prefix"
fi

ply2vri_path=/home/dell/link_to_urban_recon/third_party/ply2vri/ply2vri
# ramp_size: the truncate voxel distance
ramp_size=5
voxel_length=0.2

#for testing remesh
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
test_convert_path=$build_root/bin/test_vri_hash_tsdf_conversion

for input_ply_path in "$input_dir"/*.depthmesh.ply
do
  echo "$input_ply_path"
  base_filename=$(basename "$input_ply_path")
  output1=${base_filename%.ply}".rampsz-$ramp_size.vlen-$voxel_length.vri"
  output1=$output_prefix/$output1
  echo "input: ""$input_ply_path"
  echo "output to: ""$output1"
  echo "$ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1"
  $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1


# convert variable running length representation to hash map representation
# also save the meshed file for visualization
  echo "///////////////////////////////////////////"
  output_test_ply=${output1%.vri}".restored.ply"
  echo "remesh output: $output_test_ply"
  echo "$test_convert_path --rampsize $ramp_size --save_tsdf_bin --mesh_min_weight 0 --in $output1 --out $output_test_ply"
  $test_convert_path --rampsize $ramp_size --save_tsdf_bin --mesh_min_weight 0 --in $output1 --out $output_test_ply
done
