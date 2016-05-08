#!/bin/bash
build_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap
#input_ply_path=/home/dell/upload2/4-8/test_ply2tsdf2/0000001950_0000001951.depthmesh.ply
input_ply_path=/home/dell/upload2/4-8/test_ply2tsdf2/0000001950_0000001949.depthmesh.ply
input_ply_path=/home/dell/upload2/4-8/test_ply2tsdf2/0000001947_0000001948.depthmesh.ply
#input_ply_path=/ps/geiger/czhou/cars_semi_convex_hull/plys-test-conversion/car_2.ply
#input_ply_path=$1
ply2vri_path=/home/dell/link_to_urban_recon/third_party/ply2vri_color/ply2vri
#ply2vri_path=/home/dell/link_to_urban_recon/third_party/ply2vri_origin/ply2vri
test_convert_path=$build_root/bin/test_vri_hash_tsdf_conversion
# ramp_size: the truncate voxel distance
ramp_size=5
voxel_length=0.1

# run ply2vri from Washington University to convert the ply file to variable running length representation
# substituting the extenrion to .vri
output1=${input_ply_path%.ply}".no-test-color-rampsz-$ramp_size.vlen-$voxel_length.vri"
echo "outputing:"
echo "$input_ply_path"
echo "$output1"
echo "$ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1"
$ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1

# convert variable running length representation to hash map representation
# also save the meshed file for visualization
output_test_ply=${output1%.vri}".restored.ply"
echo "$output_test_ply"
echo "$test_convert_path --rampsize $ramp_size --save_tsdf_bin --mesh_min_weight 0 --in $output1 --out $output_test_ply"
$test_convert_path --rampsize $ramp_size --save_tsdf_bin --mesh_min_weight 0 --in $output1 --out $output_test_ply
