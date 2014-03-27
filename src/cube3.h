#ifndef __CASL_CUBE3_H__
#define __CASL_CUBE3_H__

#include <src/casl_types.h>
#include <src/point3.h>

/*!
 * \file Cube3.h
 * \brief A three dimensional cube and the basic functions associated
 */

/*!
 * \class Cube3
 *
 * A three dimensional cube and the basic functions associated. The algorithms are take from the JCP 190, 2003 paper
 * and the implementation was done by Chohong Min, 2006
 */

struct Cube3
{
private:
    inline void swap(double &phi1, double &phi2, double &f1, double &f2, Point3 &p1, Point3 &p2) const
    {
        double tmp_phi = phi1; phi1 = phi2; phi2 = tmp_phi;
        double tmp_f   = f1  ; f1   = f2  ; f2   = tmp_f;
        Point3 tmp_p   = p1  ; p1   = p2  ; p2   = tmp_p;
    }

    /*!
     * \brief interpolate a function f between two points
     * \param f1 the value of the function at the first point
     * \param phi1 the value of the level-set function at the first point
     * \param f2 the value of the function at the second point
     * \param phi2 the value of the level-set function at the second point
     * \return the interpolation of the function f weighed by the level-set values
     */
    inline double interpol_f(double f1, double phi1, double f2, double phi2) const
    {
        // NOTE: replace with an epsilon test
#ifdef CASL_THROWS
        if(phi2 == phi1) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
        return (f1*phi2 - f2*phi1)/(phi2-phi1);
    }

    /*!
     * \brief find the barycentre of two points
     * \param p1 the first point
     * \param phi1 the value of the level-set function at the first point
     * \param p2 the second point
     * \param phi2 the value of the level-set function at the second point
     * \return the barycentre of p1 and p2
     */
    inline Point3 interpol_p(Point3 p1, double phi1, Point3 p2, double phi2) const
    {
        // NOTE: replace with an epsilon test
#ifdef CASL_THROWS
        if(phi2 == phi1) throw std::domain_error("[CASL_ERROR]: division by zero.");
#endif
        return (p1*phi2 - p2*phi1)/(phi2-phi1);
    }

public:
    double x0,x1; // nodes
    double y0,y1;
    double z0,z1;

    /*!
       * \brief default constructor for Cube2, initialize the coordinates to zero
       */
    Cube3();

    /*!
       * \brief constructor for Cube2
       * \param x0
       * \param x1
       * \param y0
       * \param y1
       */
    Cube3(double x0, double x1, double y0, double y1, double z0, double z1);

    /*!
     * \brief interface_Area_In_Cell compute the are of the interface in the cube3
     * \param level_set_values ...
     * \return the are of the interface in the cell
     */
    double interface_Area_In_Cell( OctValue& level_set_values) const;

    /*!
     * \brief compute the volume of the Cube3 in the negative domain
     * \param level_set_values the values of the level-set function at the corners of the Cube3
     * \return the volume of the Cube3 in the negative domain
     */
    double volume_In_Negative_Domain( OctValue& level_set_values) const;

    /*!
     * \brief integrate a quantity over the negative domain in the Cube3
     * \param f the values of the quantity at the corners of the Cube3
     * \param p the values of the level-set at the corners of the Cube3
     * \return the integral of the quantity f over the negative domain (p<0) in the Cube3
     */
    double integral(const OctValue &f, const OctValue &level_set_values ) const;


    double integrate_Over_Interface(const OctValue &f, const OctValue &level_set_values ) const;
};
#endif // __CASL_CUBE3_H__
