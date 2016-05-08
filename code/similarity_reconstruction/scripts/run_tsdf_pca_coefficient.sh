#!/bin/bash
root_dir=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/pca_cars_deflation_1/
mean_tsdf_file=$root_dir"out_car_tsdf_pca_mean_template.bin"
mean_mat_file=$root_dir"out_caroutput_mean_mat.txt"
base_mat_file=$root_dir"out_caroutput_base_mat_comp4.txt"
coeff_mat_file=$root_dir"out_caroutput_coeff_mat_comp4.txt"
bbx_txt_file=$root_dir"out_car_boundingbox.txt"

tsdf_pca_coeff_bin=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/test_tsdf_pca_coefficient

output_dir=/is/ps2/czhou/3d-reconstruction/zc_tsdf_hashing/tsdf_pca_coeff_cars_deflation_3
if [ ! -d "$output_dir" ]; then
    mkdir $output_dir
fi
output_file=$output_dir/car_model.ply

echo "$tsdf_pca_coeff_bin --mean-tsdf $mean_tsdf_file --mean-mat $mean_mat_file --base-mat $base_mat_file --coeff-mat $coeff_mat_file --out $output_file --save_tsdf_bin --bbx-txt $bbx_txt_file"
$tsdf_pca_coeff_bin --mean-tsdf $mean_tsdf_file --mean-mat $mean_mat_file --base-mat $base_mat_file --coeff-mat $coeff_mat_file --out $output_file --save_tsdf_bin --bbx-txt $bbx_txt_file
