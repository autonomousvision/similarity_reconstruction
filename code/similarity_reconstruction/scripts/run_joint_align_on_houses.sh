#!/bin/bash
set -e
original_house_mesh_dir=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/
test_tsdf_align_bin=~/3d-reconstruction/zc_tsdf_hashing/test_joint_align
full_template_file=~/3d-reconstruction/zc_tsdf_hashing/house-meshes/house-poisson-1-1.restored_tsdf.bin
output_dir=~/3d-reconstruction/zc_tsdf_hashing/test_joint_align_multiple_houses_tsdf_usergrad_union_updated_3
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi

cnt=0
for i in 1 2 3 
do
    original_house_mesh_file=$original_house_mesh_dir"house"$i"-n1.ply"
    output_tsdf_restored_ply=${original_house_mesh_file%.ply}".restored.ply"
    output_tsdf_restored_bin=${original_house_mesh_file%.ply}".restored_tsdf.bin"
    echo $output_tsdf_restored_ply
    #if [ ! -f $output_tsdf_restored_bin ]; then
    #    ~/3d-reconstruction/zc_tsdf_hashing/utility/run_test_vri_hash_conversion.sh $original_house_mesh_file
    #fi
    cp -f $output_tsdf_restored_ply $output_dir"/"$(basename "$original_house_mesh_file" .ply)"-original.ply"

    in_model_newhouse[$cnt]=$output_tsdf_restored_bin
    cnt=$((cnt + 1))
done
    out_ply=$output_dir"/h-joint-align-res.ply"
    echo $out_ply
    echo ${in_model_newhouse[0]}
    $test_tsdf_align_bin --in-models ${in_model_newhouse[0]} ${in_model_newhouse[1]} ${in_model_newhouse[2]} --out $out_ply --save_tsdf_bin
