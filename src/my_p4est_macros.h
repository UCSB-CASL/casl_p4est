#ifndef MY_P4EST_MACROS_H
#define MY_P4EST_MACROS_H

#ifdef P4_TO_P8
#include <p8est.h>
#else
#include <p4est.h>
#endif

#define foreach_dimension(dim) \
  for (short dim = 0; dim<P4EST_DIM; ++dim)

#define foreach_tree(tr, p4est)\
  for (p4est_locidx_t tr = p4est->first_local_tree; tr <= (p4est_locidx_t) p4est->last_local_tree; ++tr)

#define foreach_local_node(n, nodes)\
  for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)

#define foreach_ghost_node(n, nodes)\
  for (p4est_locidx_t n = nodes->num_owned_indeps; n < (p4est_locidx_t) nodes->indep_nodes.elem_count; ++n)

#define foreach_node(n, nodes)\
  for (p4est_locidx_t n = 0; n < (p4est_locidx_t) nodes->indep_nodes.elem_count; ++n)

#define foreach_local_quad(q, tree)\
  for (p4est_locidx_t q = 0; q < (p4est_locidx_t) tree->quadrants.elem_count; ++q)

#define foreach_ghost_quad(q, ghost)\
  for (p4est_locidx_t q = 0; q < (p4est_locidx_t) ghost->ghosts.elem_count; ++q)

#define M_PARSER_ICAT(A,B) A ## B
#define M_PARSER_CAT(A,B) M_PARSER_ICAT(A,B)

#define M_PARSER_START for (short option_action = 0; option_action < 2; ++option_action)
#define M_PARSER_ADD_OPTION(cmd, type, var, ...) option_action == 0 ? cmd.add_option(M_PARSER_CAT(param_key_,var), M_PARSER_CAT(param_descr_,var)) : (void) (var = cmd.get(M_PARSER_CAT(param_key_,var), var));
#define M_PARSER_PARSE(cmd, argc, argv) if (option_action == 0) { cmd.parse(argc, argv); }
#define M_PARSER_STAGE_1 if (option_action == 1)

#define M_PARSER_DEFINE3(type, var, def, key, descr) type var = def; type * M_PARSER_CAT(param_ptr_,var) = &var; char M_PARSER_CAT(param_key_,var) [] =  key; char M_PARSER_CAT(param_descr_,var) [] = descr;
#define M_PARSER_DEFINE2(type, var, def, descr)      type var = def; type * M_PARSER_CAT(param_ptr_,var) = &var; char M_PARSER_CAT(param_key_,var) [] = #var; char M_PARSER_CAT(param_descr_,var) [] = descr;
#define M_PARSER_DEFINE1(type, var, def)             type var = def; type * M_PARSER_CAT(param_ptr_,var) = &var; char M_PARSER_CAT(param_key_,var) [] = #var; char M_PARSER_CAT(param_descr_,var) [] = #var;

#define M_PARSER_WRITE_VARIABLE(mpicomm, fich, type, var, ...) PetscFPrintf(mpicomm, fich, "%-30s  %g\n", #var, (double) var);

#endif // MY_P4EST_MACROS_H

