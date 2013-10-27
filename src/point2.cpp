#include "math.h"
#include "point2.h"
#include <src/CASL_math.h>

#ifdef P4_TO_P8
#include "my_p8est_utils.h"
#else
#include "my_p4est_utils.h"
#endif

Point2::Point2 ()
{
  x=y=0;
}

Point2::Point2 (double c1, double c2)
{
  x = c1;
  y = c2;
}

Point2::Point2 (const Point2& pt)
{
  x = pt.x;
  y = pt.y;
}

void Point2::operator=(Point2 pt)
{
  x = pt.x;
  y = pt.y;
}

bool Point2::operator ==(const Point2& pt)const
{
  return (fabs(pt.x-x)<EPS && fabs(pt.y-y)<1E-15);
}

Point2 Point2::operator-() const
{
  Point2 tmp(*this);
  tmp *= -1;
  return tmp;
}

Point2 Point2::operator+(const Point2& r) const
{
  Point2 tmp(*this);
  tmp += r;
  return tmp;
}

Point2 Point2::operator-(const Point2& r ) const
{
  Point2 tmp(*this);
  tmp -= r;
  return tmp;
}

Point2 Point2::operator*(double s) const
{
  Point2 tmp(*this);
  tmp *= s;
  return tmp;
}

Point2 Point2::operator/(double s) const
{
  Point2 tmp(*this);
  tmp /= s;
  return tmp;
}

void Point2::operator+=(const Point2& r)
{
  x += r.x;
  y += r.y;
}

void Point2::operator-=(const Point2& r)
{
  x -= r.x;
  y -= r.y;
}

void Point2::operator*=(double r)
{
  x *= r;
  y *= r;
}

void Point2::operator/=(double r)
{
  // NOTE: change r==0 to an epsilon test ?
#ifdef CASL_THROWS
  if(r == 0) throw std::domain_error("[CASL_ERROR]: Division by 0.");
#endif
  x /= r;
  y /= r;
}

double Point2::dot(const Point2& pt) const
{
  return x*pt.x + y*pt.y;
}

double Point2::norm_L2() const
{
  return sqrt(x*x + y*y);
}

double Point2::cross(const Point2& pt) const
{
  return x*pt.y - y*pt.x;
}

double Point2::sqr() const
{
  return x*x+y*y;
}

Point2 Point2::normalize() const
{
  Point2 A(*this);
  return A/norm_L2();
}

double Point2::norm_L2(const Point2& P1, const Point2& P2)

{
  return sqrt((P2.x-P1.x)*(P2.x-P1.x) + (P2.y-P1.y)*(P2.y-P1.y));
}

double Point2::curl(const Point2& P1, const Point2& P2)
{
  return P1.x*P2.y - P1.y*P2.x;
}

double Point2::cross(const Point2& P1, const Point2& P2)
{
  return P1.x*P2.y - P1.y*P2.x;
}

double Point2::area(const Point2& P1, const Point2& P2, const Point2& P3)
{
  double sum = (P2.x-P1.x)*(P3.y-P1.y) - (P3.x-P1.x)*(P2.y-P1.y);
  if(sum>0)
    return sum;
  return -sum;
}

std::ostream& operator<<(std::ostream& os, const Point2& p)
{
  os << p.x << ", " << p.y << std::endl;
  return os;
}

