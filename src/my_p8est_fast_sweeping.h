//
// Created by Im YoungMin on 5/7/20.
//

#ifndef FAST_SWEEPING_MY_P8EST_FAST_SWEEPING_H
#define FAST_SWEEPING_MY_P8EST_FAST_SWEEPING_H

#include <src/my_p4est_to_p8est.h>		// Define P4_TO_P8 so that we can compile my_p4est_fast_sweeping.cpp with
#include "my_p4est_fast_sweeping.h"		// that macro on.

/**
 * Note to myself: We must include first the header that defines P4_TO_P8 in order to correctly compile the CPP for
 * 3D (and 2D) correctly.  Otherwise, the CPP doesn't know of that macro when building the *.o file.
 */

#endif //FAST_SWEEPING_MY_P8EST_FAST_SWEEPING_H
