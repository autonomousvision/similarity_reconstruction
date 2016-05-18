#!/bin/bash
set -e

# source job pool file
. job_pool.sh

# source init file
#. init_paths.sh

# the root folder for all the data
data_root=$root_dir/

# the root folder for binaries of reconstruction programs
bin_root=$bin_dir

# the bin to convert depth maps in a folder to plys
depth2ply=$bin_root/depthmap_triangle_mesh_main

# the bin to convert ply files in a folder to vri format
ply2vri_path=$bin_root/ply2vri

# merge vri files into a single tsdf
vri_fuse=$bin_root/test_merge_vri

# the starting and ending image numbers for reconstruction
#startimg=1470
#endimg=1790
output_root=$result_root/reconstruction_baseline_$startimg"_"$endimg/
if [ ! -d $output_root ]; then
    mkdir $output_root
fi

# parameters
voxel_length=0.2
ramp_size=6
pos_truncation_limit=$(echo "$voxel_length * $ramp_size" | bc -l)
neg_truncation_limit=$(echo "$voxel_length * -5" | bc -l)  # smaller negative truncation limit

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

# added to output directories
output_suffix=$run_suffix"_"$startimg"_"$endimg

#do_depth2ply=1
#do_ply2vri=1
#do_fusevri=1

### 1. for depth2ply
dd_factor=0  # for testing depth discontinuity, 0 for no checking
edge_ratio=0
maxcamdist=$max_cam_distance  # max camera distance
margin=25  # how much of the depth maps are cropped
dep2ply_out_dir=$output_root/depthmap_meshing$output_suffix  # output dir
if [ $do_depth2ply -gt 0 ]; then
  echo "################### do depth2ply #####################"
  if [ -d $dep2ply_out_dir ]; then
    rm -r $dep2ply_out_dir
    mkdir $dep2ply_out_dir
  else
    mkdir $dep2ply_out_dir
  fi
  #echo $depth2ply --in-root $data_root --out $dep2ply_out_dir --image-prefix $image_prefix --depth-prefix $depth_prefix --param-prefix $cam_info_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin --edge_ratio $edge_ratio
  $depth2ply --in-root $data_root --out $dep2ply_out_dir --image-prefix $image_prefix --depth-prefix $depth_prefix --param-prefix $cam_info_prefix --dd_factor $dd_factor --max-camera-distance $maxcamdist --startimage $startimg --endimage $endimg --voxel_length $voxel_length --flatten --margin $margin --edge_ratio $edge_ratio
fi

# initialize the job pool
job_pool_init 24 0

### 2. for ply2vri: convert ply mesh files to volumetric representation using the public available ply2vri
vri_output_dir=$output_root/"depthmap_to_vri"$output_suffix
output_vri_suffix=.vri
if [ $do_ply2vri -gt 0 ]; then
  echo "################### do ply2vri #####################"
  ply2vri_input_dir=$dep2ply_out_dir
  if [ ! -d $vri_output_dir ]; then
    mkdir $vri_output_dir
  fi
  # get every ply file in $ply2vri_input_dir and convert to vri file
  for input_ply_path in "$ply2vri_input_dir"/*.depthmesh.ply
  do
    base_filename=$(basename "$input_ply_path")
    output_vri_path=${base_filename%.ply}$output_vri_suffix
    output_vri_path=$vri_output_dir/$output_vri_path
    echo "ply2vri input:  "$input_ply_path
    echo "ply2vri output: "$output_vri_path
    # echo $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output_vri_path
    job_pool_run nice -12 $ply2vri_path -r$voxel_length -l$ramp_size -h1 $input_ply_path $output_vri_path
  done
fi

# shut down the job pool
job_pool_shutdown

### 3. fusing vri's into one model
vri_fuse_in_dir=$vri_output_dir
vri_suffix=".depthmesh"$output_vri_suffix
fusing_output_dir=$output_root/"vri_fusing_result"$output_suffix
outprefix=$fusing_output_dir/recon-$startimg-$endimg".ply"
if [ $do_fusevri -gt 0 ]; then
  echo "################### do vri fusing ###################"
  if [ ! -d $fusing_output_dir ]; then
      mkdir $fusing_output_dir
  fi
  if [ $use_input_tsdf_file -gt 0 ]; then
      input_tsdf_file="--input-tsdf-filepath "$input_tsdf_file_path
  else
      input_tsdf_file=""
  fi
  # echo $vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truncation_limit --neg_truncation_limit $neg_truncation_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxel_length $input_tsdf_file
  $vri_fuse --in-dir $vri_fuse_in_dir --rampsize $ramp_size --out-prefix $outprefix --vri-suffix $vri_suffix --start_image $startimg --end_image $endimg --pos_truncation_limit $pos_truncation_limit --neg_truncation_limit $neg_truncation_limit --neg_full_weight_delta $negative_fullweight_delta --neg_weight_dist_thresh $negative_inflection_distance --neg_weight_thresh $negative_inflection_weight --voxel_length $voxel_length $input_tsdf_file
fi
output_dir=$fusing_output_dir
output_tsdf_file=$fusing_output_dir/recon-$startimg-$endimg"_tsdf.bin"
output_tsdf_ply_file=$fusing_output_dir/recon-$startimg-$endimg"_mesh.ply"
echo "sequence finished"
echo "Finished reconstruction from frame "$startimg" to "$endimg" for both sides."
echo "output_tsdf_file: " $output_tsdf_file
echo "output_tsdf__ply_file: " $output_tsdf_ply_file
rm -rf $dep2ply_out_dir 
rm -rf $vri_output_dir

