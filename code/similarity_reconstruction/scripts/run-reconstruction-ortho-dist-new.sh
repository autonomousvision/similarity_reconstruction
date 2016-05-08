#!/bin/bash
set -e

# source job pool file
. job_pool.sh

# source init file
#. init.sh

# the root folder for all the data
data_root=$root_dir/

# the root folder for binaries of reconstruction programs
bin_root=$bin_dir

# the bin to convert depth maps in a folder to plys
depth2ply=$bin_root/depthmap_triangle_mesh_main
#depth2ply=/home/dell/prev_recon_code/code/urban_reconstruction/build/bin/depthmap_triangle_mesh_main

# the bin to convert ply files in a folder to vri format
#ply2vri_path=$third_party_dir/ply2vri_color/ply2vri
ply2vri_path=$bin_root/ply2vri
#ply2vri_path=$third_party_dir/ply2vri_origin/ply2vri

# merge vri files into a single tsdf
vri_fuse=$bin_root/test_merge_vri

# only for testing: remesh a depth map from vri representation
# test_convert_path=$bin_root/test_vri_hash_tsdf_conversion

# the starting and ending image numbers for reconstruction
#startimg=1470
#endimg=1790
param_prefix=param_scale_4
image_prefix=img_00_scale_4
#depth_prefix=depth_00_slic_cropped_scale_4_filtered_masked
depth_prefix=depth_00_slic_cropped_scale_4_filtered
output_root=$result_root/reconstruction_closest_test-5-29-refractored
if [ ! -d $output_root ]; then
    mkdir $output_root
fi

# parameters
voxel_length=0.2
ramp_size=6
pos_truncation_limit=$(echo "$voxel_length * $ramp_size" | bc -l)
neg_truncation_limit=$(echo "$voxel_length * -5" | bc -l)


# in [fullweight_delta, 0] the weight will remain 1
negative_fullweight_delta=$(echo "-$voxel_length * 0.2 " | bc -l)

# in [inflection_distance, fullweight_delta] the weight will go from 1 to inflection_weight
negative_inflection_distance=$(echo "$voxel_length * -3" | bc -l)
negative_inflection_weight=0.05

echo "ramp_size:                    "$ramp_size
echo "pos_truncation_limit:         "$pos_truncation_limit
echo "neg_truncation_limit:         "$neg_truncation_limit
echo "negative_fullweight_delta:    "$negative_fullweight_delta
echo "negative_inflection_distance: "$negative_inflection_distance
echo "negative_inflection_weight:   "$negative_inflection_weight

#sleep 2

# added to output directories
output_suffix=$cam_suffix"-st-$startimg-ed-$endimg-vlen-$voxel_length-rampsz-$ramp_size-try1"

do_depth2ply=1
do_ply2vri=1
do_fusevri=1
#use_input_tsdf_file=1

### 1. for depth2ply
dd_factor=0  # for testing depth discontinuity, 0 for no checking
edge_ratio=0
maxcamdist=30  # max camera distance
margin=25  # how much of the depth maps are cropped
dep2ply_out_dir=$output_root/"output3-depthmap-meshing-ddfact-$dd_factor-edgeratio-$edge_ratio-margin-$margin"$output_suffix  # output dir
if [ $do_depth2ply -gt 0 ]; then
  echo "################### do depth2ply #####################"
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
  echo $depth2ply --in-root $data_root --out $dep2ply_out_dir --image-prefix $image_prefix --depth-prefix $depth_prefix --param-prefix $param_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin --edge_ratio $edge_ratio
  $depth2ply --in-root $data_root --out $dep2ply_out_dir --image-prefix $image_prefix --depth-prefix $depth_prefix --param-prefix $param_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin --edge_ratio $edge_ratio
fi

# initialize the job pool
job_pool_init 24 0

### 2. for ply2vri
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
    #echo $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1
    # job_pool_run nice -15 $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1
    echo $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1
    job_pool_run nice -12 $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output1
    #$ply2vri_path -r$voxel_length -l$ramp_size -h1 -s3 $input_ply_path $output1

    # # remeshing individual depth maps from vri files
    # # convert variable running length representation to hash map representation
    # # also save the meshed file for visualization
    # mesh_min_weight=0
    # echo "remesh individual depthmaps"
    # output_test_ply=${output1%.vri}".restored.ply"
    # echo "remesh output: $output_test_ply"
    # echo $test_convert_path --rampsize $ramp_size --mesh_min_weight $mesh_min_weight --in $output1 --out $output_test_ply
    # $test_convert_path --rampsize $ramp_size --mesh_min_weight $mesh_min_weight --in $output1 --out $output_test_ply
  done
fi

# shut down the job pool
 job_pool_shutdown

### 3. for fusing vri into one model
if [ $do_fusevri -gt 0 ]; then

  echo "################### do vri fusing #####################"
  vri_fuse_in_dir=$vri_output_dir
  vri_suffix=".depthmesh"$output_vri_suffix
  fusing_output_dir=$output_root/"vri-fusing-result-s3"$output_suffix
  if [ ! -d $fusing_output_dir ]; then
    mkdir $fusing_output_dir
  fi
  outprefix=$fusing_output_dir/recon-$startimg-$endimg-vlen-$voxel_length-rampsz-$ramp_size".ply"
  if [ $use_input_tsdf_file -gt 0 ]; then
      input_tsdf_file="--input-tsdf-filepath "$input_tsdf_file_path
  else
      input_tsdf_file=""
  fi

  echo $vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truncation_limit --neg_truncation_limit $neg_truncation_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxel_length $input_tsdf_file
  $vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truncation_limit --neg_truncation_limit $neg_truncation_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxel_length $input_tsdf_file
fi
output_tsdf_file=$fusing_output_dir/recon-$startimg-$endimg-vlen-$voxel_length-rampsz-$ramp_size"_tsdf.bin"
echo "cam 2 finished"

echo $outprefix
echo "start image" $startimg
echo "end image" $endimg
echo "output_tsdf_file" $output_tsdf_file

#exit


