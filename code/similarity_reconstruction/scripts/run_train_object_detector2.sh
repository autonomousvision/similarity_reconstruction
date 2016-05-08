#!/bin/bash
# the binary file of reconstruction program
test_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/bin/test_train_object_detector2

# input_model
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-3d-model-semantic/ply-0-600-0.5-newtest-conf-flatten-1-noclean_bin_tsdf_file.bin
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-debug-newdata5/ply-1890-2090-0-newtest-noconf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin
input_model=/home/dell/upload2/4-10/reconstruction/closest2/merge_vri_recon2-1890-2090-rampsz-5_tsdf.bin
#input_model=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/output-newdata1-satur-conf-cam23-new1/ply-642-1043-0-newtest-conf-flatten-0-noclean-voxellen-0.2-posmaxdist-1.6-negmaxdist--.8_bin_tsdf_file.bin

# input_sample
detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_building2.txt
#detected_obb_file=/home/dell/Data/download_labels/test_label_obb_txt/001822_002032_house_origin_detection_debug.txt

# output_prefix
set -e
vx=9
vy=9
vz=6
trynum=15
output_prefix=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/hashmap/results/tsdf_train_svm_test_newmerge_1890_2090_nonnoisy_recon_test-$trynum-nojitter-vxidelen-$vx-$vy-$vz/
if [ ! -d "$output_prefix" ]; then
mkdir $output_prefix
fi
output_prefix=$output_prefix"/house_newtsdf1"

echo $test_bin $input_samples --in-model $input_model --detected-obb-file $detected_obb_file --out-dir-prefix $output_prefix --save_tsdf_bin --sample_num 1000 --jitter_num 90 --mesh-min-weight 0.0 --sample_voxel_sidelengths $vx $vy $vz
$test_bin $input_samples --in-model $input_model --detected-obb-file $detected_obb_file --out-dir-prefix $output_prefix --save_tsdf_bin --sample_num 1000 --jitter_num 0 --mesh-min-weight 0.0 --sample_voxel_sidelengths $vx $vy $vz
