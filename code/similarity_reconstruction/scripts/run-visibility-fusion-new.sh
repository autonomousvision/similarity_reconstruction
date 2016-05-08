#!/bin/bash
# the root folder for all the data
data_root=/home/dell/Data/data_1/2013_05_28_drive_0000_sync/image_03/rect/
# the binary file of reconstruction program
bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/visibility-fusion-main
# the starting and ending image numbers for reconstruction
startimg=801
endimg=1301
# inside data_root
depth_prefix=depth_00
rgb_prefix=img_00
output_prefix=simplefusion-spherr-supp-0.5-newdata1-801-1301-2
max_cam_dist=35
support_thresh=0.5
param_prefix=param2/

set -e

echo $bin $data_root --startimage $startimg --endimage $endimg --depthprefix $depth_prefix --rgbprefix $rgb_prefix --outputprefix $output_prefix --max-camera-distance $max_cam_dist --support-thresh $support_thresh 
#echo "$bin $data_root --startimage $startimg --endimage $endimg --depthprefix $depth_prefix --rgbprefix $rgb_prefix --outputprefix $output_prefix --max-camera-distance $max_cam_dist --support-thresh $support_thresh "
$bin $data_root --startimage $startimg --endimage $endimg --depthprefix $depth_prefix --rgbprefix $rgb_prefix --outputprefix $output_prefix --max-camera-distance $max_cam_dist --support-thresh $support_thresh --paramprefix $param_prefix
