# Makefile for a dummy application
# The P4EST_DIR environment variable must point to a p4est installation

include $(P4EST_DIR)/etc/Makefile.p4est.mk
P4EST_CXX = mpicxx

TARGETS = dummy
OBJECTS = dummymain.o dummytest.o my_p4est_tools.o

default: $(TARGETS)

my_p4est_tools.o: ../../src/my_p4est_tools.c
	$(P4EST_CC) $(P4EST_CFLAGS) $(P4EST_CPPFLAGS) -c $< -o $@

.c.o:
	$(P4EST_CC) $(P4EST_CFLAGS) $(P4EST_CPPFLAGS) -c $< -o $@

.cpp.o:
	$(P4EST_CXX) $(P4EST_CFLAGS) $(P4EST_CPPFLAGS) -c $< -o $@

dummy: $(OBJECTS)
	$(P4EST_CXX) $(P4EST_CFLAGS) $(P4EST_LDFLAGS) -o $@ $^ $(P4EST_LIBS)

clean:
	rm -f *.o *.a $(OBJECTS) $(TARGETS)

realclean: clean
	rm -f *vtu *.visit

.PHONY: default clean realclean
