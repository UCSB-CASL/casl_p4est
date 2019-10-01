clear all; close all; clc;
% set the two following environment variables accordingly to your machine's
% and run's exportation paths before using the script: the first path must
% point to your local installation of petsc, the second path must point to
% the folder where the results of the executable have been exported
petsc_root = getenv('PETSC_DIR');
if isempty(petsc_root)
    error("you need to defined the environement variable PETSC_DIR before running this script");
end
output_folder = getenv('OUTPUT_FOLDER');
if isempty(output_folder)
    error("you need to defined the environement variable OUTPUT_FOLDER before running this script");
end
addpath(convertStringsToChars(strcat(petsc_root, '/share/petsc/matlab')))
filename = convertStringsToChars(strcat(output_folder, '/coordinates.mat'));
coordinates = PetscBinaryRead(filename);
filename = convertStringsToChars(strcat(output_folder, '/my_smooth_function.mat'));
smooth_fun = PetscBinaryRead(filename);

filename = convertStringsToChars(strcat(output_folder, '/second_derivative_of_smooth_function.mat'));
second_derivative = PetscBinaryRead(filename);

filename = convertStringsToChars(strcat(output_folder, '/first_derivative_of_smooth_function.mat'));
first_derivative = PetscBinaryRead(filename);

filename = convertStringsToChars(strcat(output_folder, '/first_derivative_of_smooth_function_on_new_grid.mat'));
first_derivative_on_new_grid = PetscBinaryRead(filename);


FONTSIZE = 32;
% just the function
figure('Units','normalized','Position',[0.01 0.35 0.45 0.6]);
plot(coordinates, smooth_fun, 'b-', 'linewidth', 3, 'Markersize', 20)
xlabel('$x$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
title('Node-sampled function', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
grid on
set(gca,'fontsize',FONTSIZE);
% second derivatives
figure('Units','normalized','Position',[0.48 0.35 0.45 0.6]);
plot(coordinates, second_derivative, 'b-', 'linewidth', 3, 'Markersize', 20)
xlabel('$x$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
title('Second derivatives (standard finite differences)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
grid on
set(gca,'fontsize',FONTSIZE);
% first derivatives
figure('Units','normalized','Position',[0.2 0.02 0.45 0.6]);
plot(coordinates, first_derivative, 'b-', 'linewidth', 3, 'Markersize', 20)
xlabel('$x$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
title('First derivatives (compact finite differences)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
grid on
set(gca,'fontsize',FONTSIZE);
% first derivatives on new grid
figure('Units','normalized','Position',[0.2 0.52 0.45 0.6]);
plot(coordinates, first_derivative_on_new_grid, 'r-', 'linewidth', 3, 'Markersize', 20)
xlabel('$x$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
title('First derivatives remapped', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
grid on
set(gca,'fontsize',FONTSIZE);



