#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H


#include<iostream>
#include<sstream>
#include<fstream>
#include <cstring>
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <stdio.h>
#include <time.h>



class RandomGenerator
{
private:

    int *random_integer;

public:

    time_t rawtime;
    struct tm * timeinfo;
    int t;
    double lowest;
    double highest;
    double range;
    double r;
    int rSize;

    RandomGenerator(double l,double h,int rsize);
    void continue2Generate();
    ~RandomGenerator();
    int random_Integer(int i);
    void print_All();
};

#endif // RANDOM_GENERATOR_H
