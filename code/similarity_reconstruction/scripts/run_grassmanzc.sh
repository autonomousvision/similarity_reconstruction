#~/bin/bash

grassman_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/grassman/bin/grassman_average_main

input_mat_file=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/test_rpca_res_cpp_1/1_house_extend_only_inliers_forcpp.mat

output_mat_file=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/test_rpca_res_cpp_1/output.mat

data_var_name=data_mat
weight_var_name=weight_mat

num_component=1
percentage_to_trim=1

echo $grassman_bin --input_mat_file $input_mat_file --output_mat_file $output_mat_file --data_var_name $data_var_name --weight_var_name $weight_var_name --num_component $num_component --percentage_to_trim $percentage_to_trim --alsologtostderr
$grassman_bin --input_mat_file $input_mat_file --output_mat_file $output_mat_file --data_var_name $data_var_name --weight_var_name $weight_var_name --num_component $num_component --percentage_to_trim $percentage_to_trim --alsologtostderr
