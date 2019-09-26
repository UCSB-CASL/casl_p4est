clear all;
close all;
clc;
 
% nprocs              = 80;
% min_lvl             = 6;
% max_lvl             = 9;
% nprocs              = 160;
% min_lvl             = 7;
% max_lvl             = 10;
nprocs              = 240;
min_lvl             = 7;
max_lvl             = 11;
n_fields            = [1 2 4 8 16];
exec_times          = zeros(3, length(n_fields));
base_name = sprintf('pod_test_%d_procs_min_lvl_%d_max_lvl_%d', nprocs, min_lvl, max_lvl);

for k = 1:length(n_fields)
    nn = n_fields(k);
    file_name = sprintf('%s_method_%d_n_fields_%d', base_name, 0, nn);
    [exec_times(1, k), err_grad_ref, err_second_derivatives_ref] = read_results(file_name);
    for method = 1:2
        file_name = sprintf('%s_method_%d_n_fields_%d', base_name, method, nn);
        [exec_times(method+1, k), err_grad, err_second_derivatives] = read_results(file_name);
        if (max(max(abs(err_grad - err_grad_ref))) > 0.000001*max(max(abs(err_grad_ref))))
            warning_msg = sprintf('Inconsistency to more than 0.000001 found between method %d and reference (method 0) \nfor the calculation of gradient of %d fields on a %d/%d grid with %d procs', method, nn, min_lvl, max_lvl, nprocs);
            warning(warning_msg);
        end
        if (max(max(abs(err_second_derivatives - err_second_derivatives_ref))) > 0.000001*max(max(abs(err_second_derivatives_ref))))
            warning_msg = sprintf('Inconsistency to more than 0.000001 found between method %d and reference (method 0) \nfor the calculation of second derivatives of %d fields on a %d/%d grid with %d procs', method, nn, min_lvl, max_lvl, nprocs);
            warning(warning_msg);
        end
    end
end

plot_results(nprocs, min_lvl, max_lvl, n_fields, exec_times);
