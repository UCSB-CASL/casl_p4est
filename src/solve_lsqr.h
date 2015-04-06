#ifndef SOLVE_LSQR_H
#define SOLVE_LSQR_H

#include <vector>

using std::vector;

double solve_lsqr_system(double **A, vector<double> &p, int nb_x, int nb_y, char order=2);

#endif /* SOLVE_LSQR_H */
