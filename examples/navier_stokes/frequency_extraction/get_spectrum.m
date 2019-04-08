% file_to_load = "";
% example:
file_to_load = "forces_4-7_split_threshold_0.01_cfl_1.00_sl_2.dat";
fid = fopen(file_to_load, 'r');
fgetl(fid);
read_data = fscanf(fid, '%f %f %f %f', [4 Inf]);
time = read_data(1, :)';
F = read_data(2:4, :)';
fclose(fid);

% play with these fot your file, cause it's result-dependent
cutoff_freq = 5.0;
clip_start  = 68.25;
clip_end    = 378.9;
start_avg   = 0.0, %84.62;

close all
plot_signals(time, F)
[filtered_F, avg] = filter_signals(time, F, cutoff_freq, start_avg);
fprintf('Averaged drag force = %g\n', avg(1))
fprintf('Averaged lift force = %g\n', avg(2))

name_export = "filtered_"+file_to_load;
fid = fopen(name_export, 'w');
fprintf(fid, "%% tn | Cd_x | Cd_y | Cd_z\n");
fprintf(fid, '%12.8f %12.8f %12.8f %12.8f\n', [time'; filtered_F']);
fclose(fid);
plot_signals(time, filtered_F)
[clipped_time, clipped_F] = clip_and_shift_signals(time, filtered_F, clip_start, clip_end);
[uniform_clipped_F, uniform_clipped_time] = resample(clipped_F-ones(size(clipped_F, 1), 1)*clipped_F(1, :), clipped_time, 1000, 'linear');
uniform_clipped_F = uniform_clipped_F + ones(size(uniform_clipped_F, 1), 1)*clipped_F(1, :);
% [pseudoperiodic_time, pseudoperiodic_F] = duplicate_signals(clipped_time, clipped_F);
% plot_signals(pseudoperiodic_time, pseudoperiodic_F)
% [uniform_clipped_F, uniform_clipped_time] = resample(pseudoperiodic_F-ones(size(pseudoperiodic_F, 1), 1)*pseudoperiodic_F(1, :), pseudoperiodic_time, 1000, 'linear');
% uniform_clipped_F = uniform_clipped_F + ones(size(uniform_clipped_F, 1), 1)*pseudoperiodic_F(1, :);

plot_signals(uniform_clipped_time, uniform_clipped_F)
spectrum = fft([uniform_clipped_F(:, 1) sqrt(uniform_clipped_F(:, 2).^2 + uniform_clipped_F(:, 3).^2)]);
plot_spectrum(1.0/(uniform_clipped_time(2)-uniform_clipped_time(1)), spectrum)