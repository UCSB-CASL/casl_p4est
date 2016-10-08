#ifndef SHAPES_H
#define SHAPES_H

#include <math.h>

#ifdef P4_TO_P8
class flower_phi_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;

  flower_phi_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y, double z) const
  {
    double X = x-xc;
    double Y = y-yc;
    double Z = z-zc;
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*(r-r0 - beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.)*cos(0.5*PI*Z/r0));
  }
};

class flower_phi_x_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;

  flower_phi_x_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y, double z) const
  {
    double X = x-xc;
    double Y = y-yc;
    double Z = z-zc;
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0)*cos(0.5*PI*Z/r0);
  }
};

class flower_phi_y_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;

  flower_phi_y_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y, double z) const
  {
    double X = x-xc;
    double Y = y-yc;
    double Z = z-zc;
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.)*cos(0.5*PI*Z/r0);
  }
};

class flower_phi_z_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;

  flower_phi_z_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y, double z) const
  {
    double X = x-xc;
    double Y = y-yc;
    double Z = z-zc;
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*Z*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        +inside*0.5*PI/r0*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.)*sin(0.5*PI*Z/r0);
  }
};
struct flower_shaped_domain_t
{
  flower_phi_t phi;
  flower_phi_x_t phi_x;
  flower_phi_y_t phi_y;
  flower_phi_z_t phi_z;

  flower_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    phi.set_params(r0, xc, yc, zc, beta, inside);
    phi_x.set_params(r0, xc, yc, zc, beta, inside);
    phi_y.set_params(r0, xc, yc, zc, beta, inside);
    phi_z.set_params(r0, xc, yc, zc, beta, inside);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1)
  {
    phi.set_params(r0, xc, yc, zc, beta, inside);
    phi_x.set_params(r0, xc, yc, zc, beta, inside);
    phi_y.set_params(r0, xc, yc, zc, beta, inside);
    phi_z.set_params(r0, xc, yc, zc, beta, inside);
  }
};
#else
class flower_phi_t : public CF_2
{
public:
  double r0;
  double xc, yc;
  double beta;
  double inside;

  flower_phi_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y) const
  {
    double X = x-xc;
    double Y = y-yc;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*(r-r0 - beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.));
  }
};

class flower_phi_x_t: public CF_2
{
public:
  double r0;
  double xc, yc;
  double beta;
  double inside;

  flower_phi_x_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y) const
  {
    double X = x-xc;
    double Y = y-yc;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0);
  }
};

class flower_phi_y_t: public CF_2
{
public:
  double r0;
  double xc, yc;
  double beta;
  double inside;

  flower_phi_y_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside) {}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
  }

  double operator()(double x, double y) const
  {
    double X = x-xc;
    double Y = y-yc;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.);
  }
};

struct flower_shaped_domain_t
{
  flower_phi_t phi;
  flower_phi_x_t phi_x;
  flower_phi_y_t phi_y;

  flower_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
  {
    phi.set_params(r0, xc, yc, beta, inside);
    phi_x.set_params(r0, xc, yc, beta, inside);
    phi_y.set_params(r0, xc, yc, beta, inside);
  }


  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1)
  {
    phi.set_params(r0, xc, yc, beta, inside);
    phi_x.set_params(r0, xc, yc, beta, inside);
    phi_y.set_params(r0, xc, yc, beta, inside);
  }
};
#endif

#endif // SHAPES_H
