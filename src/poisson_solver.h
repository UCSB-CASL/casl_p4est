#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <p4est.h>
#include <p4est_bits.h>
#include <p4est_mesh.h>
#include "ArrayV.h"
#include "utilities.h

#include <petsc.h>

#include "utils.h"
#include "neighbors.h"

using std::cout;
using std::endl;

class PoissonSolver
{
  const CF_2 *uex, *f;
  p4est_t *p4est;

  Mat a;
  KSP ksp;
  Vec x, b, xex;
  PetscErrorCode ierr;

  parStopWatch w;

  CellNeighbors *cell_ngbds;

public:
  PoissonSolver(p4est_t* p4est_, const CF_2& uex_, const CF_2& f_);
  ~PoissonSolver();

  void setUpNegativeLaplaceMatrix();
  void setUpNegativeLaplaceRhsVec();
  void setUpNegativeLaplaceSystem(){ setUpNegativeLaplaceMatrix(); setUpNegativeLaplaceRhsVec(); }
  void solve(Vec& sol, Vec& sol_ex);
  void save(const std::string& filename);
  void load(const std::string& filename);
};


#endif // POISSON_SOLVER_H
