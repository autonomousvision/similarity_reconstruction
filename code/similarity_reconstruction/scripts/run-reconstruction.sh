#!/bin/bash
# run the reconstruction program
# but if a tsdf bin file the same as the output name already exists, the program loads in the tsdf bin file and only do smoothing

# the root folder for all the data
data_root=/ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/
# the folder where (outlier filtered) depth maps are stored, has to be inside $data_root
depth_pref=simplefusion-spherr-supp-0_5-newdata1-keep-saturate-801-1301
depth_pref=simplefusion-spherr-supp-0_5-newdata1-801-1301
# the binary file of reconstruction program
recon_bin=../../../urban_reconstruction_build/hashmap/bin/fisheye_depths_fusion
# the starting and ending image numbers for reconstruction
startimg=0
endimg=200
# output directory
output_root=../../../urban_reconstruction_build/hashmap/
outputdir=$output_root"output-3d-model-no-saturate/"
mkdir $outputdir
set -e

# weight: the minimum allowed weight for meshing
# niteration: how many times the diffusion smoothing is executed
for weight in 0.5
do
for niteration in 1
do
outputfile_conf=$outputdir/ply-$startimg-$endimg-$weight-newtest-conf-flatten-$niteration-noclean
outputfile_conf_tsdfbin=$outputdir/ply-$startimg-$endimg-0.5-newtest-conf-flatten-1-noclean
$recon_bin --in-root /ps/geiger/czhou/2013_05_28/2013_05_28_drive_0000_sync/image_02/rect/ --out $outputfile_conf".ply" --depth-prefix $depth_pref --save-ascii --startimage $startimg --endimage $endimg --flatten --min_weight $weight --diffusion-smooth --niteration $niteration --tsdf-filepath $outputfile_conf_tsdfbin"_bin_tsdf_file.bin" --use_confidence 
done
done
