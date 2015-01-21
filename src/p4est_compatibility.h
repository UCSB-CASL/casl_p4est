#ifndef P4EST_COMPATIBILITY_H
#define P4EST_COMPATIBILITY_H

#include <p4est_config.h>

// Following is adopted from P4EST -- c.f. petscversion.h for more details

// version is equal to (MAJOR.MINOR)
#define P4EST_VERSION_IE(MAJOR,MINOR)       \
	(P4EST_VERSION_MAJOR 	 == (MAJOR) && 			\
		(P4EST_VERSION_MINOR == (MINOR)	))

// version is less than (MAJOR.MINOR)
#define P4EST_VERSION_LT(MAJOR,MINOR)       \
  (P4EST_VERSION_MAJOR    <  (MAJOR) ||    \
  	(P4EST_VERSION_MAJOR  == (MAJOR) &&    \
     P4EST_VERSION_MINOR  <  (MINOR) ))

// version is less than or equal to (MAJOR.MINOR)
#define P4EST_VERSION_LE(MAJOR,MINOR) \
  (P4EST_VERSION_LT(MAJOR,MINOR) || 	\
   P4EST_VERSION_IE(MAJOR,MINOR) )

// version is greater than (MAJOR.MINOR)
#define P4EST_VERSION_GT(MAJOR,MINOR) \
  (0 == P4EST_VERSION_LE(MAJOR,MINOR))

// version is greater than or equal (MAJOR.MINOR)
#define P4EST_VERSION_GE(MAJOR,MINOR) \
  (0 == P4EST_VERSION_LT(MAJOR,MINOR))

#endif // P4EST_COMPATIBILITY_H