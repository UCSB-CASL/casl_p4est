#include "math.h"
#include "point3.h"

#include <src/casl_math.h>

Point3::Point3 ()
{
    x=y=z=0;
}

Point3::Point3 (double c1, double c2, double c3)
{
    x = c1;
    y = c2;
    z = c3;
}

Point3::Point3(const double xyz[3]): Point3(xyz[0], xyz[1], xyz[2]) {}

Point3::Point3 (const Point2& pt)
{
	x = pt.x;
	y = pt.y;
	z = 0;
}

void Point3::operator=(const Point3& P)
{
    x = P.x;
    y = P.y;
    z = P.z;
}

Point3 Point3::operator-() const
{
    Point3 out;
    out.x = -x;
    out.y = -y;
    out.z = -z;
    return out;
}

Point3 Point3::operator+(const Point3& r) const
{
    Point3 out;
    out = *this;
    out += r;
    return out;
}

Point3 Point3::operator-(const Point3& r) const
{
    Point3 out;
    out = *this;
    out -= r;
    return out;
}

Point3 Point3::operator*(double s) const
{
    Point3 out;
    out = *this;
    out *= s;
    return out;
}

Point3 Point3::operator/(double s) const
{
    Point3 out;
    out = *this;
    out /= s;
    return out;
}

void Point3::operator+=(const Point3& r)
{
    x += r.x;
    y += r.y;
    z += r.z;
}

void Point3::operator-=(const Point3& r)
{
	x -= r.x;
	y -= r.y;
	z -= r.z;
}

void Point3::operator*=(double r)
{
	x *= r;
	y *= r;
	z *= r;
}

void Point3::operator/=(double r)
{
    // NOTE: change s==0 to an epsilon test ?
#ifdef CASL_THROWS
        if(r == 0) throw std::domain_error("[CASL_ERROR]: Division by 0.");
#endif
    x /= r;
    y /= r;
    z /= r;
}

double Point3::norm_L2() const
{
    return sqrt(x*x + y*y + z*z);
}

double Point3::dot(const Point3& pt) const
{
	return x*pt.x + y*pt.y + z*pt.z;
}

Point3 Point3::cross(const Point3 &r) const
{
    Point3 out;

    out.x = y * r.z - z * r.y;
    out.y = z * r.x - x * r.z;
    out.z = x * r.y - y * r.x;

    return out;
}

Point3 Point3::normalize() const
{
    Point3 A(*this);
    return A/norm_L2();
}

Point3 Point3::curl(const Point3& r) const
{
    return curl(*this,r);
}

Point3 Point3::curl(const Point3& l, const Point3& r)
{
	Point3 out;

	out.x = l.y * r.z - l.z * r.y;
    out.y = l.z * r.x - l.x * r.z;
	out.z = l.x * r.y - l.y * r.x;

	return out;
}

void Point3::curl(const Point3& l,
                  const Point3& r,
                  Point3& out)
{
	out.x = l.y * r.z - l.z * r.y;
    out.y = l.z * r.x - l.x * r.z;
	out.z = l.x * r.y - l.y * r.x;
}

double Point3::area(const Point3& P0,
                    const Point3& P1,
                    const Point3& P2)
{
    Point3 P01 = P1-P0;
    Point3 P02 = P2-P0;

    double area = .5*curl(P01,P02).norm_L2();

    return ABS(area);
}

double Point3::volume(const Point3& P0,
                      const Point3& P1,
                      const Point3& P2,
                      const Point3& P3)
{
    double a11 = P1.x-P0.x; double a12 = P1.y-P0.y; double a13 = P1.z-P0.z;
    double a21 = P2.x-P0.x; double a22 = P2.y-P0.y; double a23 = P2.z-P0.z;
    double a31 = P3.x-P0.x; double a32 = P3.y-P0.y; double a33 = P3.z-P0.z;

    double vol = a11*(a22*a33-a23*a32)
               + a21*(a32*a13-a12*a33)
               + a31*(a12*a23-a22*a13);

    if(vol>0) return vol/6.;
    return -vol/6.;
}

std::ostream& operator<<(std::ostream& os, const Point3& p)
{
    os << p.x << ", " << p.y << ", " << p.z << std::endl;
    return os;
}

void Point3::print()const
{
    std::cout << x << ", " << y << ", " << z << std::endl;
}
