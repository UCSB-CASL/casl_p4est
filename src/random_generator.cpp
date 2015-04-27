#include "random_generator.h"

RandomGenerator::RandomGenerator(double l, double h, int rsize)
 {
     time_t tempT=  time ( NULL );
     this->timeinfo=localtime(&tempT);
     this->t=this->timeinfo->tm_sec;
     srand(this->t);
     this->lowest=l; this->highest=h;
     this->range=this->highest-this->lowest+1;
     this->rSize=rsize;
     this->random_integer=new int[this->rSize];
     for(int index=0; index<this->rSize; index++)
     {
         this->r=rand();
         this->random_integer[index] = this->lowest+double(this->range*this->r/(RAND_MAX + 1.0));
     }
 }

 void RandomGenerator::continue2Generate()
 {
     for(int index=0; index<this->rSize; index++)
     {
         this->r=rand();
         this->random_integer[index] = this->lowest+double(this->range*this->r/(RAND_MAX + 1.0));
     }
 }

 RandomGenerator::~RandomGenerator()
 {
     if(this->random_integer!=NULL)
         delete [] this->random_integer;

 }

 int RandomGenerator::random_Integer(int i)
 {

     if(i<this->rSize && i>-1)
         return this->random_integer[i];
     else
         return -1;
 }

 void RandomGenerator::print_All()
 {
     for(int i=0;i<this->rSize;i++)
         std::cout<<this->t<<" "<<i<<" "<<this->random_Integer(i)<<std::endl;
 }
