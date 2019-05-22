#!/bin/sh
gnuplot ./tex_forces.gnu
latex ./force_history.tex
dvipdf -dAutoRotatePages=/None ./force_history.dvi
