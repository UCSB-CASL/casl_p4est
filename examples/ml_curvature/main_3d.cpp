#include <src/my_p4est_to_p8est.h>
#include <iostream>
//#include "main_2d.cpp"

int main()
{
#ifdef P4_TO_P8
	std::cout << "Test in 3D" << std::endl;
#else
	std::cerr << "P4_TO_P8 undefined!" << std::endl;
#endif
	return 0;
}