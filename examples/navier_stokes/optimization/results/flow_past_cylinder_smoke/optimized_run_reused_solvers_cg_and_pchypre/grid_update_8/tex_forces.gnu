set term epslatex color standalone
set output 'force_history.tex'
set key at 400.0,1.0 Right 
set xlabel "$t$"
set ylabel "$\\mathbf{C}_{\\mathrm{D}} = \\frac{1}{\\rho r u_{0}^{2}}\\int_{\\Gamma}{ \\left( -p \\mathbf{I} + 2\\mu \\mathbf{D} \\right)\\cdot \\mathbf{n} \\, \\mathrm{d}\\Gamma}$ " 
set xrange [300.0:400.0]
set yrange [-1.0:1.5]
plot	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" using 1:2 title '$C_{\mathrm{D}, x}$' with lines lw 3,\
	 "forces_4-6_split_threshold_0.10_cfl_1.00_sl_2.dat" using 1:3 title '$C_{\mathrm{D}, y}$' with lines lw 3
