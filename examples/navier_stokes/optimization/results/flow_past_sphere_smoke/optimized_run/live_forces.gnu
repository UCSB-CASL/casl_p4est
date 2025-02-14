set term wxt noraise
set key at 200.0,0.4  Left font "Arial,14"
set xlabel "Time" font "Arial,14"
set ylabel "Nondimensional force " font "Arial,14"
set xrange [00.0:200.0]
set yrange [0.0:0.6]
plot	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" every ::6 using 1:2 title 'x-component' with lines lw 3,\
	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" every ::6 using 1:3 title 'y-component' with lines lw 3,\
	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" every ::6 using 1:4 title 'z-component' with lines lw 3
pause 4
reread
