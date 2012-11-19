# Makefile for a dummy application
# The P4EST_DIR environment variable must point to a p4est installation

include $(P4EST_DIR)/etc/Makefile.p4est.mk

TARGETS = dummy
OBJECTS = dummymain.o dummytest.o

default: $(TARGETS)

.c.o:
	$(P4EST_CC) $(P4EST_CFLAGS) $(P4EST_CPPFLAGS) -c $< -o $@

.cpp.o:
	$(P4EST_CXX) $(P4EST_CFLAGS) $(P4EST_CPPFLAGS) -c $< -o $@

dummy: $(OBJECTS)
	$(P4EST_CXX) $(P4EST_CFLAGS) $(P4EST_LDFLAGS) -o $@ $^ $(P4EST_LIBS)

clean:
	rm -f *.o *.a $(OBJECTS) $(TARGETS)

.PHONY: default clean
