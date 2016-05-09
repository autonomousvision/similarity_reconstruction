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

if [ $do_consistency_check -gt 0 ]; then
    echo $consist_bin --in_model $scene_model --detect_obb_file $detect_obb_file --output_filename $out_prefix --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
    $consist_bin --in_model $scene_model --detect_obb_file $detect_obb_file --output_filename $out_prefix --data_roots $data_roots --cam_info_prefix $cam_info_prefix --skymap_prefix $skymap_prefix --start_image $startimg --end_image $endimg --mesh_min_weight $mesh_min_weight --sky_map_thresh $sky_thresh --max_cam_distance $max_cam_distance --depth_prefix $depth_prefix $clean_tsdf --depthmap_check $depthmap_check --skymap_check $skymap_check --filter_noise $filter_noise
fi
consistent_tsdf_output=$out_prefix".tsdf_consistency_cleaned_tsdf.bin"
consistent_tsdf_output_ply=$out_prefix".tsdf_consistency_cleaned.ply"
visualization_txt=$consistency_check_root/visualization/visualization.txt
