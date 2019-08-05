set term wxt noraise
set key top right Left font "Arial,14"
set xlabel "Time" font "Arial,14"
set ylabel "Nondimensional force " font "Arial,14"
plot	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" every 8::48 using 1:2 title 'x-component' with lines lw 3,\
	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" every 8::48 using 1:3 title 'y-component' with lines lw 3
pause 4
reread
