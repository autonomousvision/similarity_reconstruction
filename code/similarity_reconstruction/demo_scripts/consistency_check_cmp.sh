#!/bin/bash
set -e
consist_bin=$bin_dir/test_consistency_check
out_prefix=$consistency_check_root/res
if [ $check_tsdf -gt 0 ]; then
    clean_tsdf="--clean_tsdf"
else
    clean_tsdf=""
fi
sky_thresh=3
apply_to_category="0 1 2"
max_cam_distance=30
st_neighbor=-1
ed_neighbor=2
depthmap_check=1
skymap_check=1
filter_noise=60
in_obb=$out_prefix".oldobb.txt"

if [ $do_consistency_check -gt 0 ]; then
    #echo $consist_bin --in_model $scene_model --detect_obb_file $detect_obb_file --output_filename $out_prefix --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
    #sleep 1
    $consist_bin --in_model $scene_model --detect_obb_file $detect_obb_file --output_filename $out_prefix --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
    # inserted old code for comparison
    #$consist_bin --in-model $scene_model --in-obb $in_obb --out $out_prefix --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh_old --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
fi
#####################
#startim=$startimg
#endim=$endimg
#mesh_min_weight=0
#sky_thresh_old=3
#apply_to_category="0 1 2"
#max_cam_distance=30
#st_neighbor=-1
#ed_neighbor=2
#depthmap_check=1
#skymap_check=1
#filter_noise=60
#in_obb=$out_prefix".oldobb.txt"
#consist_bin_old=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/backup/test_consistency_check
#consist_outroot_old=$consistency_check_root"/oldresult/"
#if [ ! -d $consist_outroot_old ]; then
#    mkdir $consist_outroot_old
#fi
##$consist_bin_old --in-model $in_model --in-obb $in_obb --out $out --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startim --end_image $endim --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
#echo $consist_bin_old --in-model $scene_model --in-obb $in_obb --out $consist_outroot_old --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh_old --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
#$consist_bin_old --in-model $scene_model --in-obb $in_obb --out $consist_outroot_old --data-roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_thresh $sky_thresh_old --apply_to_category $apply_to_category --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --st_neighbor $st_neighbor --ed_neighbor $ed_neighbor --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
#####################
consistent_tsdf_output=$out_prefix".tsdf_consistency_cleaned_tsdf.bin"
consistent_tsdf_output_ply=$out_prefix".tsdf_consistency_cleaned.ply"
visualization_txt=$consistency_check_root/visualization/visualization.txt
