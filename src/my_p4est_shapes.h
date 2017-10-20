#ifndef SHAPES_H
#define SHAPES_H

#include <math.h>

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
//    double X = x-xc;
//    double Y = y-yc;
//    double Z = z-zc;
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
//    double X = x-xc;
//    double Y = y-yc;
//    double Z = z-zc;
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
//    double X = x-xc;
//    double Y = y-yc;
//    double Z = z-zc;
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
//    double X = x-xc;
//    double Y = y-yc;
//    double Z = z-zc;
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
//    double X = x-xc;
//    double Y = y-yc;
    double X = (x-xc)*cos_theta-(y-yc)*sin_theta;
    double Y = (x-xc)*sin_theta+(y-yc)*cos_theta;
    double r = sqrt(X*X + Y*Y);
    if (r < 1.0E-9) r = 1.0E-9; // to avoid division by zero
    return inside*(r-r0 - beta*(pow(Y,5.)+5.*pow(X,4.)*Y-10.*pow(X*Y,2.)*Y)/pow(r,5.));
//    return inside*(r-r0 - beta*cos(5.*atan2(X,Y)));
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
//    double X = x-xc;
//    double Y = y-yc;
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
//    double X = x-xc;
//    double Y = y-yc;
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

#endif // SHAPES_H
