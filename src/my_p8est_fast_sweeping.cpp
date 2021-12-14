//
// Created by Im YoungMin on 5/7/20.
//

#include <src/my_p4est_to_p8est.h>
#include "my_p4est_fast_sweeping.cpp"

/**
 * Note to myself: When the compiler finds an #include directive, it does a "copy/paste" operation.  This is the reason
 * for two fundamental things.
 * 1. We don't need to add headers to the CMake's `add_executable` as these will be are *included* in the invoker.
 * 2. We can do inclusion of full CPPs as above without the need for adding those CPP source files to `add_executable`.
 */
