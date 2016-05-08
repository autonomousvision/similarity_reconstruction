#!/bin/bash

startimg_arr=(1550 1890 3320 6050)
endimg_arr=(1650 2090 3530 6350)
startimg_arr=(1470 1890 3320 6050)
endimg_arr=(1790 2090 3530 6350)

##startimg_arr=(1470 1890 3320 6050)
##endimg_arr=(1790 1920 3530 6350)

# root folder of binary files
bin_dir=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/
bin_dir=/home/dell/prev_recon_code/code/urban_reconstruction/build/bin/
bin_dir=/home/dell/codebase/mpi_project_git/similarity_reconstruction/code/similarity_reconstruction-build/bin/

# folder containing third party software
#third_party_dir=/home/dell/link_to_urban_recon/third_party/
third_party_dir=/home/dell/prev_recon_code/code/urban_reconstruction/third_party/

#result_root=/home/dell/link_to_results/
result_root=/home/dell/results_5/
if [ ! -d $result_root ]; then
    mkdir $result_root
fi
# directory which contains the '2013_05_28' folder

#for i in 0 1 2 3
for i in 0
do
    startimg=${startimg_arr[$i]}
    endimg=${endimg_arr[$i]}
    echo $startimg
    echo $endimg
    sleep 1

    ### camera 2
    root_dir=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_02/rect
    #root_dir=/home/dell/Data/newdata_2/rect/
    cam_suffix="_cam_2"
    use_input_tsdf_file=0
    . ./run-reconstruction-ortho-dist-new.sh

    ### camera 3
    root_dir=/home/dell/Data/data-4-10/2013_05_28_drive_0000_sync/image_03/rect
    cam_suffix="_cam_3_with_2"
    use_input_tsdf_file=1
    input_tsdf_file_path=$output_tsdf_file
    . ./run-reconstruction-ortho-dist-new.sh
done
