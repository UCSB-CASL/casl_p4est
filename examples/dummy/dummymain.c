
#include <stdio.h>
#include <p4est_connectivity.h>

int main (int argc, char ** argv)
{
	p4est_connectivity_t * conn;

	printf ("Hello world\n");

	conn = p4est_connectivity_new_brick (1, 1, 0, 0);
	p4est_connectivity_destroy (conn);

	return 0;
}
