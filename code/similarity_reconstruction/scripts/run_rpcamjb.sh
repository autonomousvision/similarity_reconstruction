#~/bin/bash

rpca_bin=/home/dell/codebase/mpi_project/urban_reconstruction/code/build/rpca_mjb/bin/rpca_mjb_main

input_mat_file=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/test_rpca_res_cpp_1/1_house_extend_only_inliers_forcpp.mat

output_mat_file=/home/dell/codebase/mpi_project/urban_reconstruction/code/matlab_code/robust_pca/test_rpca_res_cpp_1/output_rpcamjb1.mat

data_var_name=data_mat
weight_var_name=weight_mat

num_component=1
#percentage_to_trim=1
data_dim_size=" 51 51 51 "

$rpca_bin --input_mat_file $input_mat_file --output_mat_file $output_mat_file --data_var_name $data_var_name --weight_var_name $weight_var_name --num_component $num_component --data_dim_size $data_dim_size --alsologtostderr
