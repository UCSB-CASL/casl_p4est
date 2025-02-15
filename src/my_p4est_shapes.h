#ifndef SHAPES_H
#define SHAPES_H

#include <math.h>
#ifdef P4_TO_P8
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_utils.h>
#endif

//--------------------------------------------------------------
// half-space
//--------------------------------------------------------------
#ifdef P4_TO_P8

class half_space_t {
public:

  class phi_t: public CF_3 {
  public:
    double nx, ny, nz;
    double x0, y0, z0;

    phi_t(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
      : nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->x0 = x0;
      this->y0 = y0;
      this->z0 = z0;
    }
    double operator()(double x, double y, double z) const
    {
      return nx*(x-x0)/sqrt(nx*nx+ny*ny+nz*nz) + ny*(y-y0)/sqrt(nx*nx+ny*ny+nz*nz) + nz*(z-z0)/sqrt(nx*nx+ny*ny+nz*nz);
    }
  } phi;

  class phi_x_t: public CF_3 {
  public:
    double nx, ny, nz;
    double x0, y0, z0;

    phi_x_t(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
      : nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->x0 = x0;
      this->y0 = y0;
      this->z0 = z0;
    }

    double operator()(double x, double y, double z) const
    {
      return nx/sqrt(nx*nx+ny*ny+nz*nz);
    }
  } phi_x;

  class phi_y_t: public CF_3 {
  public:
    double nx, ny, nz;
    double x0, y0, z0;

    phi_y_t(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
      : nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->x0 = x0;
      this->y0 = y0;
      this->z0 = z0;
    }

    double operator()(double x, double y, double z) const
    {
      return ny/sqrt(nx*nx+ny*ny+nz*nz);
    }
  } phi_y;

  class phi_z_t: public CF_3 {
  public:
    double nx, ny, nz;
    double x0, y0, z0;

    phi_z_t(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
      : nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->x0 = x0;
      this->y0 = y0;
      this->z0 = z0;
    }

    double operator()(double x, double y, double z) const
    {
      return nz/sqrt(nx*nx+ny*ny+nz*nz);
    }
  } phi_z;

  half_space_t(double nx = 0., double ny = 1., double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
  {
      phi.set_params(nx,ny,nz,x0,y0,z0);
    phi_x.set_params(nx,ny,nz,x0,y0,z0);
    phi_y.set_params(nx,ny,nz,x0,y0,z0);
    phi_z.set_params(nx,ny,nz,x0,y0,z0);
  }

  void set_params(double nx = 0., double ny = 1., double nz = 0., double x0 = 0., double y0 = 0., double z0 = 0.)
  {
      phi.set_params(nx,ny,nz,x0,y0,z0);
    phi_x.set_params(nx,ny,nz,x0,y0,z0);
    phi_y.set_params(nx,ny,nz,x0,y0,z0);
    phi_z.set_params(nx,ny,nz,x0,y0,z0);
  }

  void set_params_points(double xa, double ya, double za,
                         double xb, double yb, double zb,
                         double xc, double yc, double zc)
  {
    double x0 = xa;
    double y0 = ya;
    double z0 = za;

    double n1x = xb-xa;
    double n1y = yb-ya;
    double n1z = zb-za;

    double n2x = xc-xa;
    double n2y = yc-ya;
    double n2z = zc-za;

    double nx = n1y*n2z - n1z*n2y;
    double ny = n1z*n2x - n1x*n2z;
    double nz = n1x*n2y - n1y*n2x;

      phi.set_params(nx,ny,nz,x0,y0,z0);
    phi_x.set_params(nx,ny,nz,x0,y0,z0);
    phi_y.set_params(nx,ny,nz,x0,y0,z0);
    phi_z.set_params(nx,ny,nz,x0,y0,z0);
  }

};
#else
class half_space_t {
public:

  class phi_t: public CF_2 {
  public:
    double nx, ny;
    double x0, y0;

    phi_t(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
      : nx(nx), ny(ny), x0(x0), y0(y0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->x0 = x0;
      this->y0 = y0;
    }

    double operator()(double x, double y) const
    {
      return nx*(x-x0)/sqrt(nx*nx+ny*ny) + ny*(y-y0)/sqrt(nx*nx+ny*ny);
    }
  } phi;

  class phi_x_t: public CF_2 {
  public:
    double nx, ny;
    double x0, y0;

    phi_x_t(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
      : nx(nx), ny(ny), x0(x0), y0(y0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->x0 = x0;
      this->y0 = y0;
    }

    double operator()(double x, double y) const
    {
      (void) x; (void) y;
      return nx/sqrt(nx*nx+ny*ny);
    }
  } phi_x;

  class phi_y_t: public CF_2 {
  public:
    double nx, ny;
    double x0, y0;

    phi_y_t(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
      : nx(nx), ny(ny), x0(x0), y0(y0) {}

    void set_params(double nx = 0.0, double ny = 1.0, double x0 = 0., double y0 = 0.)
    {
      this->nx = nx;
      this->ny = ny;
      this->x0 = x0;
      this->y0 = y0;
    }

    double operator()(double x, double y) const
    {
      (void) x; (void) y;
      return ny/sqrt(nx*nx+ny*ny);
    }
  } phi_y;

  half_space_t(double nx = 0., double ny = 1., double x0 = 0., double y0 = 0.)
  {
      phi.set_params(nx,ny,x0,y0);
    phi_x.set_params(nx,ny,x0,y0);
    phi_y.set_params(nx,ny,x0,y0);
  }

  void set_params(double nx = 0., double ny = 1., double x0 = 0., double y0 = 0.)
  {
      phi.set_params(nx,ny,x0,y0);
    phi_x.set_params(nx,ny,x0,y0);
    phi_y.set_params(nx,ny,x0,y0);
  }

  void set_params_points(double xa, double ya, double xb, double yb)
  {
    double nx =-(yb-ya);
    double ny = (xb-xa);
    double x0 = xa;
    double y0 = ya;

      phi.set_params(nx,ny,x0,y0);
    phi_x.set_params(nx,ny,x0,y0);
    phi_y.set_params(nx,ny,x0,y0);
  }

};
#endif

//--------------------------------------------------------------
// flower-shaped domain
//--------------------------------------------------------------
#ifdef P4_TO_P8
class flower_phi_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;
  double theta;
  double nx,ny,nz;
  double R[9];

  flower_phi_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
               double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
    : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside),
      nx(nx_), ny(ny_), nz(nz_), theta(theta)
  {
    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);
    R[1] = nx*ny*(1.-ct)-nz*st;
    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st;
    R[4] = ct+ny*ny*(1.-ct);
    R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st;
    R[7] = nz*ny*(1.-ct)+nx*st;
    R[8] = ct+nz*nz*(1.-ct);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                  double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;
    this->theta = theta;

    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  double operator()(double x, double y, double z) const
  {
    double X = R[0]*(x-xc)+R[1]*(y-yc)+R[2]*(z-zc);
    double Y = R[3]*(x-xc)+R[4]*(y-yc)+R[5]*(z-zc);
    double Z = R[6]*(x-xc)+R[7]*(y-yc)+R[8]*(z-zc);
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
  double theta;
  double nx,ny,nz;
  double R[9];

  flower_phi_x_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                 double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
      : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside),
    nx(nx_), ny(ny_), nz(nz_), theta(theta)
  {
    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);
    R[1] = nx*ny*(1.-ct)-nz*st;
    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st;
    R[4] = ct+ny*ny*(1.-ct);
    R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st;
    R[7] = nz*ny*(1.-ct)+nx*st;
    R[8] = ct+nz*nz*(1.-ct);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                  double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;
    this->theta = theta;

    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  double operator()(double x, double y, double z) const
  {
    double X = R[0]*(x-xc)+R[1]*(y-yc)+R[2]*(z-zc);
    double Y = R[3]*(x-xc)+R[4]*(y-yc)+R[5]*(z-zc);
    double Z = R[6]*(x-xc)+R[7]*(y-yc)+R[8]*(z-zc);
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0)*cos(0.5*PI*Z/r0);
    double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.)*cos(0.5*PI*Z/r0);
    double phi_z = inside*Z*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        +inside*0.5*PI/r0*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.)*sin(0.5*PI*Z/r0);
    return phi_x*R[0]+phi_y*R[3]+phi_z*R[6];
  }
};

class flower_phi_y_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;
  double theta;
  double nx,ny,nz;
  double R[9];

  flower_phi_y_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                 double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
      : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside),
    nx(nx_), ny(ny_), nz(nz_), theta(theta)
  {
    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);
    R[1] = nx*ny*(1.-ct)-nz*st;
    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st;
    R[4] = ct+ny*ny*(1.-ct);
    R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st;
    R[7] = nz*ny*(1.-ct)+nx*st;
    R[8] = ct+nz*nz*(1.-ct);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                  double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;
    this->theta = theta;

    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  double operator()(double x, double y, double z) const
  {
    double X = R[0]*(x-xc)+R[1]*(y-yc)+R[2]*(z-zc);
    double Y = R[3]*(x-xc)+R[4]*(y-yc)+R[5]*(z-zc);
    double Z = R[6]*(x-xc)+R[7]*(y-yc)+R[8]*(z-zc);
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0)*cos(0.5*PI*Z/r0);
    double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.)*cos(0.5*PI*Z/r0);
    double phi_z = inside*Z*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        +inside*0.5*PI/r0*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.)*sin(0.5*PI*Z/r0);
    return phi_x*R[1]+phi_y*R[4]+phi_z*R[7];
  }
};

class flower_phi_z_t: public CF_3
{
public:
  double r0;
  double xc, yc, zc;
  double beta;
  double inside;
  double theta;
  double nx,ny,nz;
  double R[9];

  flower_phi_z_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                 double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
      : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside),
    nx(nx_), ny(ny_), nz(nz_), theta(theta)
  {
    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);
    R[1] = nx*ny*(1.-ct)-nz*st;
    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st;
    R[4] = ct+ny*ny*(1.-ct);
    R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st;
    R[7] = nz*ny*(1.-ct)+nx*st;
    R[8] = ct+nz*nz*(1.-ct);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                  double nx_ = 0, double ny_ = 0, double nz_ = 1, double theta = 0.0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->zc=zc;
    this->beta=beta;
    this->inside=inside;

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;
    this->theta = theta;

    double ct = cos(theta);
    double st = sin(theta);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  double operator()(double x, double y, double z) const
  {
    double X = R[0]*(x-xc)+R[1]*(y-yc)+R[2]*(z-zc);
    double Y = R[3]*(x-xc)+R[4]*(y-yc)+R[5]*(z-zc);
    double Z = R[6]*(x-xc)+R[7]*(y-yc)+R[8]*(z-zc);
    double r = sqrt(X*X + Y*Y + Z*Z);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0)*cos(0.5*PI*Z/r0);
    double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.)*cos(0.5*PI*Z/r0);
    double phi_z = inside*Z*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.)*cos(0.5*PI*Z/r0))/r
        +inside*0.5*PI/r0*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.)*sin(0.5*PI*Z/r0);
    return phi_x*R[2]+phi_y*R[5]+phi_z*R[8];
  }
};
struct flower_shaped_domain_t
{
  flower_phi_t phi;
  flower_phi_x_t phi_x;
  flower_phi_y_t phi_y;
  flower_phi_z_t phi_z;

  flower_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                         double nx = 0, double ny = 0, double nz = 1, double theta = 0.0)
  {
    phi.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_x.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_y.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_z.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double beta = 0, double inside = 1,
                  double nx = 0, double ny = 0, double nz = 1, double theta = 0.0)
  {
    phi.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_x.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_y.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
    phi_z.set_params(r0, xc, yc, zc, beta, inside, nx, ny, nz, theta);
  }
};
#else
class flower_phi_t : public CF_2
{
public:
  double r0;      // radius of the undeformed circle
  double xc, yc;  // center
  double beta;    // degree of the deformation of the circle
  double inside;  // exterior (-1) or interior (-1)
  double theta, cos_theta, sin_theta;   // rotational angle and auxiliary variables

  flower_phi_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta) {cos_theta = cos(theta); sin_theta = sin(theta);}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
    this->theta=theta; cos_theta = cos(theta); sin_theta = sin(theta);
  }

  double operator()(double x, double y) const
  {
    double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
    double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
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
  double theta, cos_theta, sin_theta;   // rotational angle and auxiliary variables

  flower_phi_x_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta) {cos_theta = cos(theta); sin_theta = sin(theta);}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
    this->theta=theta; cos_theta = cos(theta); sin_theta = sin(theta);
  }

  double operator()(double x, double y) const
  {
    double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
    double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0);
    double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.);
    return phi_x*cos_theta+phi_y*sin_theta;
  }
};

class flower_phi_y_t: public CF_2
{
public:
  double r0;
  double xc, yc;
  double beta;
  double inside;
  double theta, cos_theta, sin_theta;   // rotational angle and auxiliary variables

  flower_phi_y_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
    : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta) {cos_theta = cos(theta); sin_theta = sin(theta);}

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
  {
    this->r0=r0;
    this->xc=xc;
    this->yc=yc;
    this->beta=beta;
    this->inside=inside;
    this->theta=theta; cos_theta = cos(theta); sin_theta = sin(theta);
  }

  double operator()(double x, double y) const
  {
    double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
    double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    double phi_x = inside*X*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*20.*beta*X*Y*(X*X-Y*Y)/pow(r,5.0);
    double phi_y = inside*Y*(1. + 5.*beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,6.))/r
        -inside*5.*beta*(pow(Y,4.)+pow(X,4.)-6.*pow(X*Y,2.))/pow(r,5.);
    return -phi_x*sin_theta+phi_y*cos_theta;
  }
};

struct flower_shaped_domain_t
{
  flower_phi_t phi;
  flower_phi_x_t phi_x;
  flower_phi_y_t phi_y;

  flower_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
  {
    phi.set_params(r0, xc, yc, beta, inside, theta);
    phi_x.set_params(r0, xc, yc, beta, inside, theta);
    phi_y.set_params(r0, xc, yc, beta, inside, theta);
  }


  void set_params(double r0 = 1, double xc = 0, double yc = 0, double beta = 0, double inside = 1, double theta = 0)
  {
    phi.set_params(r0, xc, yc, beta, inside, theta);
    phi_x.set_params(r0, xc, yc, beta, inside, theta);
    phi_y.set_params(r0, xc, yc, beta, inside, theta);
  }
};
#endif

//--------------------------------------------------------------
// Arbitrary radially perturbed spherical domain
//--------------------------------------------------------------
#ifdef P4_TO_P8
class radial_phi_t : public CF_3
{
  double p = 4;
public:
  double r0;      // radius of the undeformed circle
  double xc, yc, zc;  // center
  double inside;  // exterior (-1) or interior (-1)
  int N;          // number of perturbations
  std::vector<double> n;      // mode
  std::vector<double> beta;   // degree of the deformation of the circle
  std::vector<double> theta;  // rotational angle and auxiliary variables
  cf_value_type_t what;
  double rot;
  double nx,ny,nz;
  double R[9];

  radial_phi_t(cf_value_type_t what, double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double inside = 1,
               int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL,
               double nx=0, double ny=0, double nz=1, double rot=0)
  {
    this->what = what;
    set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx, ny, nz, rot);
  }

  radial_phi_t(cf_value_type_t what, double r0, double xc, double yc, double zc, double inside,
               double n, double beta, double theta,
               double nx=0, double ny=0, double nz=1, double rot=0)
  {
    this->what = what;
    set_params(r0, xc, yc, zc, inside, n, beta, theta, nx, ny, nz, rot);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double inside = 1,
                  int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL,
                  double nx_=0, double ny_=0, double nz_=1, double rot=0)
  {
    this->r0     = r0;
    this->xc     = xc;
    this->yc     = yc;
    this->zc     = zc;
    this->inside = inside;
    this->N      = N;

    this->n.resize(N);
    this->beta.resize(N);
    this->theta.resize(N);

    for (int i = 0; i < N; ++i)
    {
      this->n[i]      = n[i];
      this->beta[i]   = beta[i];
      this->theta[i]  = theta[i];
    }

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;

    double ct = cos(rot);
    double st = sin(rot);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  void set_params(double r0, double xc, double yc, double zc, double inside,
                  double n, double beta, double theta,
                  double nx_=0, double ny_=0, double nz_=1, double rot=0)
  {
    this->r0     = r0;
    this->xc     = xc;
    this->yc     = yc;
    this->zc     = zc;
    this->inside = inside;
    this->N      = 1;

    this->n.assign(1, n);
    this->beta.assign(1, beta);
    this->theta.assign(1, theta);

    this->nx = nx_;
    this->ny = ny_;
    this->nz = nz_;

    double ct = cos(rot);
    double st = sin(rot);
    double norm = sqrt(nx*nx+ny*ny+nz*nz);
    this->nx /= norm;
    this->ny /= norm;
    this->nz /= norm;
    R[0] = ct+nx*nx*(1.-ct);    R[1] = nx*ny*(1.-ct)-nz*st;    R[2] = nx*nz*(1.-ct)+ny*st;
    R[3] = ny*nx*(1.-ct)+nz*st; R[4] = ct+ny*ny*(1.-ct);       R[5] = ny*nz*(1.-ct)-nx*st;
    R[6] = nz*nx*(1.-ct)-ny*st; R[7] = nz*ny*(1.-ct)+nx*st;    R[8] = ct+nz*nz*(1.-ct);
  }

  double operator()(double x, double y, double z) const
  {
    double X = R[0]*(x-xc)+R[1]*(y-yc)+R[2]*(z-zc);
    double Y = R[3]*(x-xc)+R[4]*(y-yc)+R[5]*(z-zc);
    double Z = R[6]*(x-xc)+R[7]*(y-yc)+R[8]*(z-zc);

    double t = atan2(Y,X);
    double r = sqrt(X*X + Y*Y + Z*Z);
    double r2 = sqrt(X*X + Y*Y);

    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    if (r2 < 1.0E-9) r2 = 1.0E-9; // to avoid division by zero

    switch (what)
    {
      case VAL:
      {
        double perturb_0 = 0;

        for (int i = 0; i < N; ++i)
        {
          perturb_0 +=  beta[i]*cos(n[i]*(t - theta[i]));
        }

        return inside*( r -r0*(1.+ perturb_0*pow(r2/r,p)) );
      }

      case DDX:
      case DDY:
      case DDZ:
      {
        double perturb_0 = 0;
        double perturb_1 = 0;

        for (int i = 0; i < N; ++i)
        {
          perturb_0 +=  beta[i]*cos(n[i]*(t - theta[i]));
          perturb_1 +=  beta[i]*sin(n[i]*(t - theta[i]))*n[i];
        }

        double phi_x = inside*( X/r - r0*perturb_0*p*pow(r2/r,p-2.)*X*Z*Z/pow(r, 4.) - r0*Y/r2/r2*perturb_1*pow(r2/r,p)  );
        double phi_y = inside*( Y/r - r0*perturb_0*p*pow(r2/r,p-2.)*Y*Z*Z/pow(r, 4.) + r0*X/r2/r2*perturb_1*pow(r2/r,p)  );
        double phi_z = inside*( Z/r + r0*perturb_0*p*pow(r2/r,p-2.)*Z*r2*r2/pow(r, 4.) );

        switch (what) {
          case DDX: return phi_x*R[0] + phi_y*R[3] + phi_z*R[6];
          case DDY: return phi_x*R[1] + phi_y*R[4] + phi_z*R[7];
          case DDZ: return phi_x*R[2] + phi_y*R[5] + phi_z*R[8];
        }
      }

      case CUR:
      {
        double perturb_0 = 0;
        double perturb_1 = 0;
        double perturb_2 = 0;

        for (int i = 0; i < N; ++i)
        {
          perturb_0 +=  beta[i]*cos(n[i]*(t - theta[i]));
          perturb_1 += -beta[i]*sin(n[i]*(t - theta[i]))*n[i];
          perturb_2 += -beta[i]*cos(n[i]*(t - theta[i]))*n[i]*n[i];
        }

        double t_x = - Y/r/r;
        double t_y =   X/r/r;

        double t_xx = 2.*X*Y/pow(r,4.0);
        double t_xy = (Y*Y-X*X)/pow(r,4.0);
        double t_yy =-2.*X*Y/pow(r,4.0);

        double r_x = X/r;
        double r_y = Y/r;
        double r_z = Z/r;

        double r_xx = (r-X*X)/r/r;
        double r_yy = (r-Y*Y)/r/r;
        double r_zz = (r-Z*Z)/r/r;

        double r_xy = ( -X*Y)/r/r;
        double r_yz = ( -Y*Z)/r/r;
        double r_zx = ( -Z*X)/r/r;

        double r2_x = X/r2;
        double r2_y = Y/r2;
        double r2_z = 0;

        double r2_xx = (r2-X*X)/r2/r2;
        double r2_yy = (r2-Y*Y)/r2/r2;
        double r2_xy = (r2-X*Y)/r2/r2;

        double r2_zz = 0;
        double r2_yz = 0;
        double r2_zx = 0;

        double e = perturb_0;

        double e_x = perturb_1*t_x;
        double e_y = perturb_1*t_y;
        double e_z = 0;

        double e_xx = perturb_1*t_xx + perturb_2*t_x*t_x;
        double e_yy = perturb_1*t_yy + perturb_2*t_y*t_y;
        double e_xy = perturb_1*t_xy + perturb_2*t_x*t_y;

        double e_zz = 0;
        double e_yz = 0;
        double e_zx = 0;

        double psi = pow(r2/r, .5*p);

        double psi_x = .5*p*pow(r2/r, .5*p-1.)*(r2_x*r-r2*r_x)/r2/r2;
        double psi_y = .5*p*pow(r2/r, .5*p-1.)*(r2_y*r-r2*r_y)/r2/r2;
        double psi_z = .5*p*pow(r2/r, .5*p-1.)*(r2_z*r-r2*r_z)/r2/r2;

        double psi_xx = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_x*r-r2*r_x)*(r2_x*r-r2*r_x)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_xx*r + r2_x*r_x - r2_x*r_x - r2*r_xx)/pow(r,2.) - 2.*(r2_x*r-r2*r_x)*r_x/pow(r,3.));
        double psi_yy = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_y*r-r2*r_y)*(r2_y*r-r2*r_y)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_yy*r + r2_y*r_y - r2_y*r_y - r2*r_yy)/pow(r,2.) - 2.*(r2_y*r-r2*r_y)*r_y/pow(r,3.));
        double psi_zz = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_z*r-r2*r_z)*(r2_z*r-r2*r_z)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_zz*r + r2_z*r_z - r2_z*r_z - r2*r_zz)/pow(r,2.) - 2.*(r2_z*r-r2*r_z)*r_z/pow(r,3.));
        double psi_xy = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_x*r-r2*r_x)*(r2_y*r-r2*r_y)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_xy*r + r2_x*r_y - r2_y*r_x - r2*r_xy)/pow(r,2.) - 2.*(r2_x*r-r2*r_x)*r_y/pow(r,3.));
        double psi_yz = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_y*r-r2*r_y)*(r2_z*r-r2*r_z)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_yz*r + r2_y*r_z - r2_z*r_y - r2*r_yz)/pow(r,2.) - 2.*(r2_y*r-r2*r_y)*r_z/pow(r,3.));
        double psi_zx = .5*p*(.5*p-1.)*pow(r2/r, .5*p-2.)*(r2_z*r-r2*r_z)*(r2_x*r-r2*r_x)/pow(r2, 4.) - .5*p*pow(r2/r, .5*p-1.)*((r2_zx*r + r2_z*r_x - r2_x*r_z - r2*r_zx)/pow(r,2.) - 2.*(r2_z*r-r2*r_z)*r_x/pow(r,3.));

        double phi_x = inside*(r_x - r0*e_x);
        double phi_y = inside*(r_y - r0*e_y);
        double phi_z = inside*(r_z - r0*e_z);

        double phi_xx = inside*(r_xx - r0*(e_xx*psi + e_x*psi_x + e_x*psi_x + e*psi_xx));
        double phi_yy = inside*(r_yy - r0*(e_yy*psi + e_y*psi_y + e_y*psi_y + e*psi_yy));
        double phi_zz = inside*(r_zz - r0*(e_zz*psi + e_z*psi_z + e_z*psi_z + e*psi_zz));

        double phi_xy = inside*(r_xy - r0*(e_xy*psi + e_x*psi_y + e_x*psi_y + e*psi_xy));
        double phi_yz = inside*(r_yz - r0*(e_yz*psi + e_y*psi_z + e_y*psi_z + e*psi_yz));
        double phi_zx = inside*(r_zx - r0*(e_zx*psi + e_z*psi_x + e_z*psi_x + e*psi_zx));

        return (phi_x*phi_y*phi_xy - 2.*phi_x*phi_y*phi_xy + phi_x*phi_y*phi_xy +
                phi_y*phi_z*phi_yz - 2.*phi_y*phi_z*phi_yz + phi_y*phi_z*phi_yz +
                phi_z*phi_x*phi_zx - 2.*phi_z*phi_x*phi_zx + phi_z*phi_x*phi_zx)/pow(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z + EPS, 1.5);
      }
      default:
        throw;
    }
  }
};

struct radial_shaped_domain_t
{
  radial_phi_t phi;
  radial_phi_t phi_x;
  radial_phi_t phi_y;
  radial_phi_t phi_z;
  radial_phi_t phi_c;

  radial_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double inside = 1,
                         int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL,
                         double nx_=0, double ny_=0, double nz_=1, double rot=0)
    : phi(VAL), phi_x(DDX), phi_y(DDY), phi_z(DDZ), phi_c(CUR)
  {
    this->set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
  }

  radial_shaped_domain_t(double r0, double xc, double yc, double zc, double inside,
                         double n, double beta, double theta,
                         double nx_=0, double ny_=0, double nz_=1, double rot=0)
    : phi(VAL), phi_x(DDX), phi_y(DDY), phi_z(DDZ), phi_c(CUR)
  {
    this->set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
  }

  inline void set_params(double r0 = 1, double xc = 0, double yc = 0, double zc = 0, double inside = 1,
                         int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL,
                         double nx_=0, double ny_=0, double nz_=1, double rot=0)
  {
    phi  .set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
    phi_x.set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
    phi_y.set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
    phi_z.set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
    phi_c.set_params(r0, xc, yc, zc, inside, N, n, beta, theta, nx_, ny_, nz_, rot);
  }

  inline void set_params(double r0, double xc, double yc, double zc, double inside,
                         double n, double beta, double theta,
                         double nx_=0, double ny_=0, double nz_=1, double rot=0)
  {
    phi  .set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
    phi_x.set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
    phi_y.set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
    phi_z.set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
    phi_c.set_params(r0, xc, yc, zc, inside, n, beta, theta, nx_, ny_, nz_, rot);
  }
};
#else
class radial_phi_t : public CF_2
{
public:
  double r0;      // radius of the undeformed circle
  double xc, yc;  // center
  double inside;  // exterior (-1) or interior (-1)
  int N;          // number of perturbations
  std::vector<double> n;      // mode
  std::vector<double> beta;   // degree of the deformation of the circle
  std::vector<double> theta;  // rotational angle and auxiliary variables
  cf_value_type_t what;

  radial_phi_t(cf_value_type_t what, double r0 = 1, double xc = 0, double yc = 0, double inside = 1,
               int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL)
  {
    this->what = what;
    set_params(r0, xc, yc, inside, N, n, beta, theta);
  }

  radial_phi_t(cf_value_type_t what, double r0, double xc, double yc, double inside, double n, double beta, double theta)
  {
    this->what = what;
    set_params(r0, xc, yc, inside, n, beta, theta);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double inside = 1, int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL)
  {
    this->r0     = r0;
    this->xc     = xc;
    this->yc     = yc;
    this->inside = inside;
    this->N      = N;

    this->n.resize(N);
    this->beta.resize(N);
    this->theta.resize(N);

    for (int i = 0; i < N; ++i)
    {
      this->n[i]      = n[i];
      this->beta[i]   = beta[i];
      this->theta[i]  = theta[i];
    }
  }

  void set_params(double r0, double xc, double yc, double inside, double n, double beta, double theta)
  {
    this->r0     = r0;
    this->xc     = xc;
    this->yc     = yc;
    this->inside = inside;
    this->N      = 1;

    this->n.assign(1, n);
    this->beta.assign(1, beta);
    this->theta.assign(1, theta);
  }


  double operator()(double x, double y) const
  {
    double X = (x-xc);
    double Y = (y-yc);

    double t = atan2(Y,X);
    double r = sqrt(X*X + Y*Y);

    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero

    double perturb = 0;

    switch (what) {
      case VAL:
        for (int i = 0; i < N; ++i)
          perturb += beta[i]*cos(n[i]*(t - theta[i]));

        return inside*(r - r0*(1+perturb));

      case DDX:
        for (int i = 0; i < N; ++i)
          perturb += beta[i]*n[i]*sin(n[i]*(t - theta[i]));

        return inside*(X/r - r0*Y/r/r*perturb);

      case DDY:
        for (int i = 0; i < N; ++i)
          perturb += beta[i]*n[i]*sin(n[i]*(t - theta[i]));

        return inside*(Y/r + r0*X/r/r*perturb);

      case CUR:
      {
        double perturb_1 = 0;
        double perturb_2 = 0;

        for (int i = 0; i < N; ++i)
        {
          perturb_1 += -beta[i]*sin(n[i]*(t - theta[i]))*n[i];
          perturb_2 += -beta[i]*cos(n[i]*(t - theta[i]))*n[i]*n[i];
        }

        double t_x = - Y/r/r;
        double t_y =   X/r/r;

        double t_xx =    2.*X*Y/pow(r,4.0);
        double t_xy = (Y*Y-X*X)/pow(r,4.0);
        double t_yy =   -2.*X*Y/pow(r,4.0);

        double r_x = X/r;
        double r_y = Y/r;

        double r_xx = (r*r-X*X)/r/r/r;
        double r_xy = (   -X*Y)/r/r/r;
        double r_yy = (r*r-Y*Y)/r/r/r;

        double e_x = perturb_1*t_x;
        double e_y = perturb_1*t_y;

        double e_xx = perturb_1*t_xx + perturb_2*t_x*t_x;
        double e_xy = perturb_1*t_xy + perturb_2*t_x*t_y;
        double e_yy = perturb_1*t_yy + perturb_2*t_y*t_y;

        double phi_x = inside*(r_x - r0*e_x);
        double phi_y = inside*(r_y - r0*e_y);

        double phi_xx = inside*(r_xx - r0*e_xx);
        double phi_xy = inside*(r_xy - r0*e_xy);
        double phi_yy = inside*(r_yy - r0*e_yy);

        return (phi_x*phi_x*phi_yy - 2.*phi_x*phi_y*phi_xy + phi_y*phi_y*phi_xx)/pow(phi_x*phi_x + phi_y*phi_y + EPS, 1.5);
      }
      default:
        throw;
    }
  }
};

struct radial_shaped_domain_t
{
  radial_phi_t phi;
  radial_phi_t phi_x;
  radial_phi_t phi_y;
  radial_phi_t phi_c;

  radial_shaped_domain_t(double r0 = 1, double xc = 0, double yc = 0, double inside = 1, int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL)
    : phi(VAL), phi_x(DDX), phi_y(DDY), phi_c(CUR)
  {
    this->set_params(r0, xc, yc, inside, N, n, beta, theta);
  }

  radial_shaped_domain_t(double r0, double xc, double yc, double inside, double n, double beta, double theta)
    : phi(VAL), phi_x(DDX), phi_y(DDY), phi_c(CUR)
  {
    this->set_params(r0, xc, yc, inside, n, beta, theta);
  }

  void set_params(double r0 = 1, double xc = 0, double yc = 0, double inside = 1, int N = 0, double *n = NULL, double *beta = NULL, double *theta = NULL)
  {
    phi  .set_params(r0, xc, yc, inside, N, n, beta, theta);
    phi_x.set_params(r0, xc, yc, inside, N, n, beta, theta);
    phi_y.set_params(r0, xc, yc, inside, N, n, beta, theta);
    phi_c.set_params(r0, xc, yc, inside, N, n, beta, theta);
  }

  void set_params(double r0, double xc, double yc, double inside, double n, double beta, double theta)
  {
    phi  .set_params(r0, xc, yc, inside, n, beta, theta);
    phi_x.set_params(r0, xc, yc, inside, n, beta, theta);
    phi_y.set_params(r0, xc, yc, inside, n, beta, theta);
    phi_c.set_params(r0, xc, yc, inside, n, beta, theta);
  }
};
#endif


class capsule_phi_t : public CF_DIM
{
public:
  double r0;      // radius
  double DIM(xc, yc, zc);  // center
  double l;       // length
  double inside;  // exterior (-1) or interior (-1)
  double theta;   // rotational angle (around z) and auxiliary variables
  cf_value_type_t what;

  capsule_phi_t(cf_value_type_t what, double r0 = 1, DIM(double xc = 0, double yc = 0, double zc = 0), double l = 0, double inside = 1, double theta = 0)
  {
    this->what = what;
    set_params(r0, DIM(xc, yc, zc), l, inside, theta);
  }

  void set_params(double r0 = 1, DIM(double xc = 0, double yc = 0, double zc = 0), double l = 0, double inside = 1, double theta = 0)
  {
    this->r0     = r0;
    EXECD( this->xc = xc, this->yc = yc, this->zc = zc);
    this->inside = inside;
    this->l      = l;
    this->theta  = theta;
  }

  double operator()(DIM(double x, double y, double z)) const
  {
    double DIM( X = (x-xc)*cos(theta)-(y-yc)*sin(theta),
                Y = (x-xc)*sin(theta)+(y-yc)*cos(theta),
                Z = z);

    double X0 = MAX(-.5*l, MIN(.5*l, X));

    double r = ABSD(X-X0, Y, Z);

    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero

    switch (what) {
      case VAL: return inside*(r-r0);
      case DDX: return inside*(X-X0)/r;
      case DDY: return inside*Y/r;
#ifdef P4_TO_P8
      case DDZ: return inside*Z/r;
#endif
      case CUR:
      {
        if (X > -.5*l && X < .5*l) return 0;
        else return inside*1./r;
      }
      default: throw;
    }
  }
};

struct capsule_domain_t
{
  capsule_phi_t phi, phi_c, DIM( phi_x, phi_y, phi_z );

  capsule_domain_t(double r0 = 1, DIM(double xc = 0, double yc = 0, double zc = 0), double l = 0, double inside = 1, double theta = 0)
    : phi(VAL), phi_c(CUR), DIM(phi_x(DDX), phi_y(DDY), phi_z(DDZ))
  {
    this->set_params(r0, DIM(xc, yc, zc), l, inside, theta);
  }

  void set_params(double r0 = 1, DIM(double xc = 0, double yc = 0, double zc = 0), double l = 0, double inside = 1, double theta = 0)
  {
    phi  .set_params(r0, DIM(xc, yc, zc), l, inside, theta);
    phi_c.set_params(r0, DIM(xc, yc, zc), l, inside, theta);
    EXECD( phi_x.set_params(r0, DIM(xc, yc, zc), l, inside, theta),
           phi_y.set_params(r0, DIM(xc, yc, zc), l, inside, theta),
           phi_z.set_params(r0, DIM(xc, yc, zc), l, inside, theta) );
  }
};


// Added By Faranak for multi shapes
#ifdef P4_TO_P8
// 3D version of multi_circle classes
class multi_circle_phi_t : public CF_3 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc, zc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                      const std::vector<double>& xc = std::vector<double>{0.0},
                      const std::vector<double>& yc = std::vector<double>{0.0},
                      const std::vector<double>& zc = std::vector<double>{0.0},
                      double beta = 0.0, double inside = 1.0, double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& zc,
                   double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->zc = zc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y, double z) const {
        if (r0.empty()) return 0.0;

        double phi_final = evaluate_sphere(x, y, z, 0);
        for(size_t i = 1; i < r0.size(); ++i) {
            if(inside > 0) {
                phi_final = std::min(phi_final, evaluate_sphere(x, y, z, i));
            } else {
                phi_final = std::max(phi_final, evaluate_sphere(x, y, z, i));
            }
        }
        return phi_final;
    }

private:
    double evaluate_sphere(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }
};

class multi_circle_phi_x_t : public CF_3 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc, zc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_x_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                       const std::vector<double>& xc = std::vector<double>{0.0},
                       const std::vector<double>& yc = std::vector<double>{0.0},
                       const std::vector<double>& zc = std::vector<double>{0.0},
                       double beta = 0.0, double inside = 1.0, double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& zc,
                   double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->zc = zc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y, double z) const {
        if (r0.empty()) return 0.0;

        size_t extremum_idx = 0;
        double extremum_val = evaluate_sphere_phi(x, y, z, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_sphere_phi(x, y, z, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_sphere_derivative_x(x, y, z, extremum_idx);
    }

private:
    double evaluate_sphere_phi(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }

    double evaluate_sphere_derivative_x(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);
        double phi_z = inside * Z / r;

        return phi_x * cos_theta + phi_y * sin_theta;
    }
};

class multi_circle_phi_y_t : public CF_3 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc, zc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_y_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                       const std::vector<double>& xc = std::vector<double>{0.0},
                       const std::vector<double>& yc = std::vector<double>{0.0},
                       const std::vector<double>& zc = std::vector<double>{0.0},
                       double beta = 0.0, double inside = 1.0, double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& zc,
                   double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->zc = zc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y, double z) const {
        if (r0.empty()) return 0.0;

        size_t extremum_idx = 0;
        double extremum_val = evaluate_sphere_phi(x, y, z, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_sphere_phi(x, y, z, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_sphere_derivative_y(x, y, z, extremum_idx);
    }

private:
    double evaluate_sphere_phi(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }

    double evaluate_sphere_derivative_y(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);
        double phi_z = inside * Z / r;

        return -phi_x * sin_theta + phi_y * cos_theta;
    }
};

class multi_circle_phi_z_t : public CF_3 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc, zc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_z_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                       const std::vector<double>& xc = std::vector<double>{0.0},
                       const std::vector<double>& yc = std::vector<double>{0.0},
                       const std::vector<double>& zc = std::vector<double>{0.0},
                       double beta = 0.0, double inside = 1.0, double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), zc(zc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& zc,
                   double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->zc = zc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y, double z) const {
        if (r0.empty()) return 0.0;

        size_t extremum_idx = 0;
        double extremum_val = evaluate_sphere_phi(x, y, z, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_sphere_phi(x, y, z, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_sphere_derivative_z(x, y, z, extremum_idx);
    }

private:
    double evaluate_sphere_phi(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }

    double evaluate_sphere_derivative_z(double x, double y, double z, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double Z = z - zc[idx];
        double r = sqrt(X*X + Y*Y + Z*Z);
        if (r < 1.0E-9) r = 1.0E-9;

        // Z derivative is simpler as sphere is symmetric in Z direction
        return inside * Z / r;
    }
};

// 3D domain struct
struct multi_circle_domain_t {
    multi_circle_phi_t phi;
    multi_circle_phi_x_t phi_x;
    multi_circle_phi_y_t phi_y;
    multi_circle_phi_z_t phi_z;

    multi_circle_domain_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         const std::vector<double>& zc = std::vector<double>{0.0},
                         double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        set_params(r0, xc, yc, zc, beta, inside, theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& zc,
                   double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        phi.set_params(r0, xc, yc, zc, beta, inside, theta);
        phi_x.set_params(r0, xc, yc, zc, beta, inside, theta);
        phi_y.set_params(r0, xc, yc, zc, beta, inside, theta);
        phi_z.set_params(r0, xc, yc, zc, beta, inside, theta);
    }
};

#else
class multi_circle_phi_t : public CF_2 {
public:
    std::vector<double> r0;      // radii of circles
    std::vector<double> xc, yc;  // centers
    double beta;                 // deformation parameter (same for all circles)
    double inside;               // interior (1) or exterior (-1)
    double theta, cos_theta, sin_theta;  // rotation parameters (same for all circles)

    multi_circle_phi_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                       const std::vector<double>& xc = std::vector<double>{0.0},
                       const std::vector<double>& yc = std::vector<double>{0.0},
                       double beta = 0.0, double inside = 1.0, double theta = 0.0)
            : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                    const std::vector<double>& xc,
                    const std::vector<double>& yc,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty() || xc.empty() || yc.empty()) return 0.0;

        double phi_final = evaluate_circle(x, y, 0);
        for(size_t i = 1; i < r0.size(); ++i) {
            if(inside > 0) {
                phi_final = std::min(phi_final, evaluate_circle(x, y, i));
            } else {
                phi_final = std::max(phi_final, evaluate_circle(x, y, i));
            }
        }
        return phi_final;
    }

private:
    double evaluate_circle(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }
};

class multi_circle_phi_x_t : public CF_2 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_x_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         double beta = 0.0, double inside = 1.0, double theta = 0.0)
            : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                    const std::vector<double>& xc,
                    const std::vector<double>& yc,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty() || xc.empty() || yc.empty()) return 0.0;

        // Find which circle gives the extremum value
        size_t extremum_idx = 0;
        double extremum_val = evaluate_circle_phi(x, y, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_circle_phi(x, y, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_circle_derivative_x(x, y, extremum_idx);
    }

private:
    double evaluate_circle_phi(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }

    double evaluate_circle_derivative_x(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                       - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                       - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);

        return phi_x * cos_theta + phi_y * sin_theta;
    }
};

class multi_circle_phi_y_t : public CF_2 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;

    multi_circle_phi_y_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         double beta = 0.0, double inside = 1.0, double theta = 0.0)
            : r0(r0), xc(xc), yc(yc), beta(beta), inside(inside), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                    const std::vector<double>& xc,
                    const std::vector<double>& yc,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->beta = beta;
        this->inside = inside;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty() || xc.empty() || yc.empty()) return 0.0;

        // Find which circle gives the extremum value
        size_t extremum_idx = 0;
        double extremum_val = evaluate_circle_phi(x, y, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_circle_phi(x, y, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_circle_derivative_y(x, y, extremum_idx);
    }

private:
    double evaluate_circle_phi(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        return inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));
    }

    double evaluate_circle_derivative_y(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                       - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                       - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);

        return -phi_x * sin_theta + phi_y * cos_theta;
    }
};

struct multi_circle_domain_t {
    multi_circle_phi_t phi;
    multi_circle_phi_x_t phi_x;
    multi_circle_phi_y_t phi_y;

    multi_circle_domain_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                          const std::vector<double>& xc = std::vector<double>{0.0},
                          const std::vector<double>& yc = std::vector<double>{0.0},
                          double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        set_params(r0, xc, yc, beta, inside, theta);
    }

    void set_params(const std::vector<double>& r0,
                    const std::vector<double>& xc,
                    const std::vector<double>& yc,
                    double beta = 0.0, double inside = 1.0, double theta = 0.0)
    {
        phi.set_params(r0, xc, yc, beta, inside, theta);
        phi_x.set_params(r0, xc, yc, beta, inside, theta);
        phi_y.set_params(r0, xc, yc, beta, inside, theta);
    }
};
#endif

// Added by Faranak partial circles
class weighted_circle_phi_t : public CF_2 {
public:
    std::vector<double> r0;         // radii
    std::vector<double> xc, yc;     // centers
    std::vector<double> split_angle; // angle where dominance changes
    double beta;                     // deformation parameter
    double inside;                   // interior (1) or exterior (-1)
    double theta, cos_theta, sin_theta;  // rotation parameters
    bool is_first_region;  // Whether this represents first region of circle

    weighted_circle_phi_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         const std::vector<double>& split = std::vector<double>{3.0*M_PI/2.0},
                         double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                         double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), split_angle(split), beta(beta), inside(inside),
          is_first_region(is_first_region), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
               const std::vector<double>& xc,
               const std::vector<double>& yc,
               const std::vector<double>& split,
               double beta = 0.0, double inside = 1.0, bool is_first_region = true,
               double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->split_angle = split;
        this->beta = beta;
        this->inside = inside;
        this->is_first_region = is_first_region;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty()) return 0.0;

        double phi_final = evaluate_circle(x, y, 0);
        for(size_t i = 1; i < r0.size(); ++i) {
            if(inside > 0) {
                phi_final = std::min(phi_final, evaluate_circle(x, y, i));
            } else {
                phi_final = std::max(phi_final, evaluate_circle(x, y, i));
            }
        }
        return phi_final;
    }



private:
    double evaluate_circle(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;

        double angle = atan2(Y, X);
        while (angle < 0) angle += 2 * M_PI;

        // Basic level set
        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        double phi = inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));

        double dist = std::abs(angle - split_angle[idx]);
        double angle_factor = exp(-dist*dist/2.0);  // Gaussian-like smoothing
        double offset = 1e-8 * r0[idx] * (r/r0[idx]) * (1.0 + angle_factor);
        if (is_first_region) {
            return angle <= split_angle[idx] ? phi : phi + offset;
        } else {
            return angle > split_angle[idx] ? phi : phi + offset;
        }
    }
};

class weighted_circle_phi_x_t : public CF_2 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc;
    std::vector<double> split_angle;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;
    bool is_first_region;

    weighted_circle_phi_x_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         const std::vector<double>& split = std::vector<double>{3.0*M_PI/2.0},
                         double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                         double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), split_angle(split), beta(beta), inside(inside),
          is_first_region(is_first_region), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& split,
                   double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                   double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->split_angle = split;
        this->beta = beta;
        this->inside = inside;
        this->is_first_region = is_first_region;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty()) return 0.0;

        // Find controlling circle
        size_t extremum_idx = 0;
        double extremum_val = evaluate_circle_phi(x, y, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_circle_phi(x, y, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_circle_derivative_x(x, y, extremum_idx);
    }

private:
    double evaluate_circle_phi(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;

        double angle = atan2(Y, X);
        while (angle < 0) angle += 2 * M_PI;

        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        double phi = inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));

        double dist = std::abs(angle - split_angle[idx]);
        double angle_factor = exp(-dist*dist/2.0);  // Gaussian-like smoothing
        double offset = 1e-8 * r0[idx] * (r/r0[idx]) * (1.0 + angle_factor);
        if (is_first_region) {
            return angle <= split_angle[idx] ? phi : phi + offset;
        } else {
            return angle > split_angle[idx] ? phi : phi + offset;
        }
    }

    double evaluate_circle_derivative_x(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;

        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);

        return phi_x * cos_theta + phi_y * sin_theta;
    }
};

class weighted_circle_phi_y_t : public CF_2 {
public:
    std::vector<double> r0;
    std::vector<double> xc, yc;
    std::vector<double> split_angle;
    double beta;
    double inside;
    double theta, cos_theta, sin_theta;
    bool is_first_region;

    // Constructor same as phi_x_t
    weighted_circle_phi_y_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                         const std::vector<double>& xc = std::vector<double>{0.0},
                         const std::vector<double>& yc = std::vector<double>{0.0},
                         const std::vector<double>& split = std::vector<double>{3.0*M_PI/2.0},
                         double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                         double theta = 0.0)
        : r0(r0), xc(xc), yc(yc), split_angle(split), beta(beta), inside(inside),
          is_first_region(is_first_region), theta(theta)
    {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& split,
                   double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                   double theta = 0.0)
    {
        this->r0 = r0;
        this->xc = xc;
        this->yc = yc;
        this->split_angle = split;
        this->beta = beta;
        this->inside = inside;
        this->is_first_region = is_first_region;
        this->theta = theta;
        this->cos_theta = cos(theta);
        this->sin_theta = sin(theta);
    }

    double operator()(double x, double y) const {
        if (r0.empty()) return 0.0;

        size_t extremum_idx = 0;
        double extremum_val = evaluate_circle_phi(x, y, 0);

        for(size_t i = 1; i < r0.size(); ++i) {
            double current = evaluate_circle_phi(x, y, i);
            if((inside > 0 && current < extremum_val) ||
               (inside < 0 && current > extremum_val)) {
                extremum_val = current;
                extremum_idx = i;
            }
        }

        return evaluate_circle_derivative_y(x, y, extremum_idx);
    }

private:
    double evaluate_circle_phi(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;

        double angle = atan2(Y, X);
        while (angle < 0) angle += 2 * M_PI;

        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;
        double phi = inside * (r - r0[idx] - beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 5.0));

        double dist = std::abs(angle - split_angle[idx]);
        double angle_factor = exp(-dist*dist/2.0);  // Gaussian-like smoothing
        double offset = 1e-8 * r0[idx] * (r/r0[idx]) * (1.0 + angle_factor);
        if (is_first_region) {
            return angle <= split_angle[idx] ? phi : phi + offset;
        } else {
            return angle > split_angle[idx] ? phi : phi + offset;
        }
    }

    double evaluate_circle_derivative_y(double x, double y, size_t idx) const {
        double X = (x - xc[idx]) * cos_theta - (y - yc[idx]) * sin_theta;
        double Y = (x - xc[idx]) * sin_theta + (y - yc[idx]) * cos_theta;

        double r = sqrt(X*X + Y*Y);
        if (r < 1.0E-9) r = 1.0E-9;

        double phi_x = inside * X * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 20.0 * beta * X * Y * (X*X - Y*Y) / pow(r, 5.0);
        double phi_y = inside * Y * (1.0 + 5.0 * beta * (pow(Y, 5.0) + 5.0 * pow(X, 4.0) * Y - 10.0 * pow(X * Y, 2.0) * Y) / pow(r, 6.0)) / r
                    - inside * 5.0 * beta * (pow(Y, 4.0) + pow(X, 4.0) - 6.0 * pow(X * Y, 2.0)) / pow(r, 5.0);

        return -phi_x * sin_theta + phi_y * cos_theta;
    }
};

struct weighted_circle_domain_t {
    weighted_circle_phi_t phi;
    weighted_circle_phi_x_t phi_x;
    weighted_circle_phi_y_t phi_y;

    weighted_circle_domain_t(const std::vector<double>& r0 = std::vector<double>{1.0},
                           const std::vector<double>& xc = std::vector<double>{0.0},
                           const std::vector<double>& yc = std::vector<double>{0.0},
                           const std::vector<double>& split = std::vector<double>{3.0*M_PI/2.0},
                           double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                           double theta = 0.0)
        : phi(r0, xc, yc, split, beta, inside, is_first_region, theta),
          phi_x(r0, xc, yc, split, beta, inside, is_first_region, theta),
          phi_y(r0, xc, yc, split, beta, inside, is_first_region, theta)
    {
        set_params(r0, xc, yc, split, beta, inside, is_first_region, theta);
    }

    void set_params(const std::vector<double>& r0,
                   const std::vector<double>& xc,
                   const std::vector<double>& yc,
                   const std::vector<double>& split,
                   double beta = 0.0, double inside = 1.0, bool is_first_region = true,
                   double theta = 0.0)
    {
        phi.set_params(r0, xc, yc, split, beta, inside, is_first_region, theta);
        phi_x.set_params(r0, xc, yc, split, beta, inside, is_first_region, theta);
        phi_y.set_params(r0, xc, yc, split, beta, inside, is_first_region, theta);
    }
};

#endif // SHAPES_H