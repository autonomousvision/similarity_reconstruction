#!/bin/bash
set -e
# the root folder for all the data
#data_root=/home/dell/Data/2013_05_28_drive_0000_sync/image_02/rect
data_root=/home/dell/Data/newdata_2/rect/
# the root folder for binaries of reconstruction programs
bin_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/
# the bin to convert depth maps in a folder to plys
depth2ply=$bin_root/depthmap_triangle_mesh_main
# the bin to convert ply files in a folder to vri format
# the bin is in the third_party/ply2vri_color folder
ply2vri_path=/home/dell/link_to_urban_recon/third_party/ply2vri_color/ply2vri
# merge vri files into a single tsdf
vri_fuse=$bin_root/test_merge_vri
# only for testing: remesh a depth map from vri representation
test_convert_path=$bin_root/test_vri_hash_tsdf_conversion

# the starting and ending image numbers for reconstruction
startimg=1470
endimg=1790
param_prefix=param_scale_4
image_prefix=img_00_scale_4
depth_prefix=depth_00_slic_cropped_scale_4_filtered_masked
#param_prefix=param
#image_prefix=img_00
#depth_prefix=simplefusion-spherr-supp-0.5-newdata1-keep-satur-crop-5
output_root=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/
# added to output directories
output_suffix="_st-$startimg-ed-$endimg-try1"
mesh_min_weight=0

do_depth2ply=1
do_ply2vri=1
do_fusevri=1

### 1. for depth2ply
voxel_length=0.5
dep2ply_out_dir=$output_root/"output-depthmap-meshing"$output_suffix  # output dir
if [ $do_depth2ply -gt 0 ]; then
echo "################### do depth2ply #####################"
dd_factor=0  # for testing depth discontinuity, 0 for no checking
maxcamdist=30  # max camera distance
margin=25  # how much of the depth maps are cropped
if [ -d $dep2ply_out_dir ]; then
rm -r $dep2ply_out_dir
mkdir $dep2ply_out_dir
else
mkdir $dep2ply_out_dir
## prompt to overwrite existing directory
#while true; do
#    read -p "$dep2ply_out_dir already exists, remove existing content?" yn
#    case $yn in
#        [Yy]* ) rm -r $dep2ply_out_dir; mkdir $dep2ply_out_dir; break;;
#        [Nn]* ) break;; #exit;;
#        * ) echo "Please answer yes or no.";;
#    esac
#done
fi
echo $depth2ply --in-root $data_root --out $dep2ply_out_dir --depth-prefix $depth_prefix --param-prefix $param_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin
$depth2ply --in-root $data_root --out $dep2ply_out_dir --image-prefix $image_prefix --depth-prefix $depth_prefix --param-prefix $param_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin
fi

### 2. for ply2vri
ramp_size=2 #truncation limit in voxels
vri_output_dir=$output_root/"output-depthmap-vri"$output_suffix
output_vri_suffix=.rampsz-"$ramp_size".vlen-"$voxel_length".vri
if [ $do_ply2vri -gt 0 ]; then
echo "################### do ply2vri #####################"
ply2vri_input_dir=$dep2ply_out_dir
if [ ! -d $vri_output_dir ]; then
mkdir $vri_output_dir
fi
# get every ply file in $ply2vri_input_dir and convert to vri file
for input_ply_path in "$ply2vri_input_dir"/*.depthmesh.ply
do
  echo "ply2vri: $input_ply_path"
  base_filename=$(basename "$input_ply_path")
  output1=${base_filename%.ply}$output_vri_suffix
  output1=$vri_output_dir/$output1
  echo "ply2vri: input is: ""$input_ply_path"
  echo "ply2vri: output to: ""$output1"
  echo $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1
  $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1

  # remeshing individual depth maps from vri files
  # convert variable running length representation to hash map representation
  # also save the meshed file for visualization
  echo "remesh individual depthmaps"
  output_test_ply=${output1%.vri}".restored.ply"
  echo "remesh output: $output_test_ply"
  echo $test_convert_path --rampsize $ramp_size --mesh_min_weight $mesh_min_weight --in $output1 --out $output_test_ply
  $test_convert_path --rampsize $ramp_size --mesh_min_weight $mesh_min_weight --in $output1 --out $output_test_ply
done
fi

### 3. for fusing vri into one model
if [ $do_fusevri -gt 0 ]; then
echo "################### do vri fusing #####################"
vri_fuse_in_dir=$vri_output_dir
vri_suffix=".depthmesh"$output_vri_suffix
fusing_output_dir=$output_root/"vri-fusing-result"$output_suffix
if [ ! -d $fusing_output_dir ]; then
mkdir $fusing_output_dir
fi
voxellen=$voxel_length
outprefix=$fusing_output_dir/merge_vri_recon2-$startimg-$endimg-rampsz-$ramp_size".ply"
pos_truc_limit=$(echo "$voxellen * $ramp_size" | bc -l)
neg_truc_limit=$(echo "$voxellen * -5" | bc -l)
# in [fullweight_delta, 0] the weight will remain 1
negative_fullweight_delta=$(echo "-$voxellen/5.0" | bc -l)
# in [inflection_distance, fullweight_delta] the weight will go from 1 to inflection_weight
negative_inflection_distance=$(echo "$voxellen * -3" | bc -l)
negative_inflection_weight=0.05
echo $pos_truc_limit
echo $neg_truc_limit
echo $negative_fullweight_delta
echo $negative_inflection_distance
echo $negative_inflection_weight

echo $vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truc_limit --neg_truncation_limit $neg_truc_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxellen
$vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truc_limit --neg_truncation_limit $neg_truc_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxellen
fi

echo "finished"
