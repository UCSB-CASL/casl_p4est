#include "cube2.h"
#include <petsclog.h>

#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

Cube2::Cube2()
{
  xyz_mmm[0] = xyz_ppp[0] = xyz_mmm[1] = xyz_ppp[1] = 0.0;
}

Cube2::Cube2(double x0, double x1, double y0, double y1)
{
  this->xyz_mmm[0] = x0; this->xyz_ppp[0] = x1;
  this->xyz_mmm[1] = y0; this->xyz_ppp[1] = y1;
}

void Cube2::kuhn_Triangulation(Simplex2& s1, Simplex2& s2 ) const
{
  s1.x0 = xyz_mmm[0] ; s1.x1 = xyz_ppp[0] ; s1.x2 = xyz_ppp[0];
  s1.y0 = xyz_mmm[1] ; s1.y1 = xyz_mmm[1] ; s1.y2 = xyz_ppp[1];

  s2.x0 = xyz_mmm[0] ; s2.x1 = xyz_mmm[0] ; s2.x2 = xyz_ppp[0];
  s2.y0 = xyz_mmm[1] ; s2.y1 = xyz_ppp[1] ; s2.y2 = xyz_ppp[1];
}

double Cube2::interface_Length_In_Cell(const QuadValue& level_set_values) const
{
  QuadValue tmp(1.,1.,1.,1.);
  return integrate_Over_Interface(tmp, level_set_values);
}

double Cube2::area_In_Negative_Domain(const QuadValue& level_set_values) const
{
  QuadValue tmp(1.,1.,1.,1.);
  return integral(tmp,level_set_values);
}

double Cube2::integral( QuadValue f ) const
{
  PetscErrorCode ierr = PetscLogFlops(8); CHKERRXX(ierr);
  return (f.val[0]+f.val[2]+f.val[1]+f.val[3])/4.*(xyz_ppp[0]-xyz_mmm[0])*(xyz_ppp[1]-xyz_mmm[1]);
}

double Cube2::integral( const QuadValue& f, const QuadValue& level_set_values ) const
{
  if     (level_set_values.val[0]<=0 && level_set_values.val[2]<=0 && level_set_values.val[1]<=0 && level_set_values.val[3]<=0 ) return integral(f);
  else if(level_set_values.val[0]> 0 && level_set_values.val[2]> 0 && level_set_values.val[1]> 0 && level_set_values.val[3]> 0 ) return 0;
  else
  {
    Simplex2 S1,S2; kuhn_Triangulation(S1,S2);

    return S1.integral(f.val[0],f.val[2],f.val[3],level_set_values.val[0],level_set_values.val[2],level_set_values.val[3])
        + S2.integral(f.val[0],f.val[1],f.val[3],level_set_values.val[0],level_set_values.val[1],level_set_values.val[3]);
  }
}

double Cube2::integrate_Over_Interface( const QuadValue& f, const QuadValue& level_set_values ) const
{
  double sum = 0.0;

  Point2 p00(xyz_mmm[0], xyz_mmm[1]); double f00 = f.val[0]; double phi00 = level_set_values.val[0];
  Point2 p01(xyz_mmm[0], xyz_ppp[1]); double f01 = f.val[1]; double phi01 = level_set_values.val[1];
  Point2 p10(xyz_ppp[0], xyz_mmm[1]); double f10 = f.val[2]; double phi10 = level_set_values.val[2];
  Point2 p11(xyz_ppp[0], xyz_ppp[1]); double f11 = f.val[3]; double phi11 = level_set_values.val[3];

  // [RAPHAEL:] I am ****PISSED****, I have wasted an entire day trying to find yet another bug due to sign
  // errors in this very function!
  //
  // I actually do not care mure about this one, except when it makes my whole set up crash on big grids on
  // Stampede and thus make things CRASH.
  //
  // I am going to introduce some *CLEAR* sign convention in here to fix my issue, I do not care if someone
  // changes it later on, but PLEASE, juste triple-check the consistency of your sign conventions!!!!!
  // Here below:
  // phi <= 0 --> negative domain
  // phi > 0 --> positive domain
  // WHATEVER CHANGE BROUGHT BY WHOMEVER CANNOT ALLOW ONE SINGLE VALUE TO BE CONSIDERED BOTH IN NEGATIVE AND
  // IN POSITIVE DOMAIN!!!

  // simple cases
  if(phi00 <= 0.0 && phi01 <= 0.0 && phi10 <= 0.0 && phi11 <= 0.0) return 0.0;
  if(phi00  > 0.0 && phi01  > 0.0 && phi10  > 0.0 && phi11  > 0.0) return 0.0;

  // iteration on each simplex in the Kuhn triangulation
  for(int n = 0; n < 2;n++)
  {
    Point2 p0 = p00; double f0 = f00; double phi0 = phi00;
    Point2 p2 = p11; double f2 = f11; double phi2 = phi11;
    // triangle (P0,P1,P2) with values (F0,F1,F2), (Phi0,Phi1,Phi2)
    Point2   p1 = (n == 0 ?   p01 :   p10);
    double   f1 = (n == 0 ?   f01 :   f10);
    double phi1 = (n == 0 ? phi01 : phi10);

    if (0)
    {
      Point2 p_00(0.5*(xyz_mmm[0] + xyz_ppp[0]),0.5*(xyz_mmm[1] + xyz_ppp[1]));
      double f_00 = 0.25*(f00 + f01 + f10 + f11);
      double l_00 = 0.25*(phi00 + phi01 + phi10 + phi11);

      Point2 p_mm(xyz_mmm[0], xyz_mmm[1]); double f_mm = f00; double l_mm = phi00;
      Point2 p_pm(xyz_ppp[0], xyz_mmm[1]); double f_pm = f10; double l_pm = phi10;
      Point2 p_mp(xyz_mmm[0], xyz_ppp[1]); double f_mp = f01; double l_mp = phi01;
      Point2 p_pp(xyz_ppp[0], xyz_ppp[1]); double f_pp = f11; double l_pp = phi11;
      switch (n) {
      case 0:
        p0    = p_00; p1  = p_mm; p2  = p_pm;
        f0    = f_00; f1  = f_mm; f2  = f_pm;
        phi0  = l_00; phi1= l_mm; phi2= l_pm; break;
      case 1:
        p0    = p_00; p1  = p_pm; p2  = p_pp;
        f0    = f_00; f1  = f_pm; f2  = f_pp;
        phi0  = l_00; phi1= l_pm; phi2= l_pp; break;
      case 2:
        p0    = p_00; p1  = p_pp; p2  = p_mp;
        f0    = f_00; f1  = f_pp; f2  = f_mp;
        phi0  = l_00; phi1= l_pp; phi2= l_mp; break;
      case 3:
        p0    = p_00; p1  = p_mp; p2  = p_mm;
        f0    = f_00; f1  = f_mp; f2  = f_mm;
        phi0  = l_00; phi1= l_mp; phi2= l_mm; break;
      }
    }

    // simple cases
    if(phi0 <=  0.0 && phi1 <=  0.0 && phi2 <=  0.0) continue;
    if(phi0  >  0.0 && phi1  >  0.0 && phi2  >  0.0) continue;

    //
    int number_of_negatives = 0;

    if(phi0 <= 0.0) number_of_negatives++;
    if(phi1 <= 0.0) number_of_negatives++;
    if(phi2 <= 0.0) number_of_negatives++;

#ifdef CASL_THROWS
    if(number_of_negatives != 1 && number_of_negatives != 2) throw std::runtime_error("[CASL_ERROR]: Wrong configuration.");
#endif

    if(number_of_negatives == 2)
    {
      phi0 *= -1.0;
      phi1 *= -1.0;
      phi2 *= -1.0;
    }

    // sorting for simplication into one case
    if(phi0 > 0 && phi1 <= 0.0) swap(phi0, phi1, f0, f1, p0, p1);
    if(phi0 > 0 && phi2 <= 0.0) swap(phi0, phi2, f0, f2, p0, p2);
    if(phi1 > 0 && phi2 <= 0.0) swap(phi1, phi2, f1, f2, p1, p2);

    // type : (-++)
    Point2 p_btw_01 = interpol_p(p0, phi0, p1, phi1); Point2 p_btw_02 = interpol_p(p0, phi0, p2, phi2);
    double f_btw_01 = interpol_f(f0, phi0, f1, phi1); double f_btw_02 = interpol_f(f0, phi0, f2, phi2);

    double length_of_line_segment = (p_btw_02 - p_btw_01).norm_L2();

    sum += length_of_line_segment * (f_btw_02 + f_btw_01)/2.;

    PetscErrorCode ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }

  return sum;
}


double Cube2::integrate_Over_Interface(const CF_2& f, const QuadValue& level_set_values ) const
{
  double sum=0;

  Point2 p00(xyz_mmm[0],xyz_mmm[1]); double f00 = 0; double phi00 = level_set_values.val[0];
  Point2 p01(xyz_mmm[0],xyz_ppp[1]); double f01 = 0; double phi01 = level_set_values.val[1];
  Point2 p10(xyz_ppp[0],xyz_mmm[1]); double f10 = 0; double phi10 = level_set_values.val[2];
  Point2 p11(xyz_ppp[0],xyz_ppp[1]); double f11 = 0; double phi11 = level_set_values.val[3];

  // simple cases
  if(phi00<=0 && phi01<=0 && phi10<=0 && phi11<=0) return 0;
  if(phi00>=0 && phi01>=0 && phi10>=0 && phi11>=0) return 0;

  // iteration on each simplex in the Kuhn triangulation
  for(int n=0;n<2;n++)
  {
    Point2 p0=p00; double f0=f00; double phi0=phi00;
    Point2 p2=p11; double f2=f11; double phi2=phi11;

    // triangle (P0,P1,P2) with values (F0,F1,F2), (Phi0,Phi1,Phi2)
    Point2   p1 = (n==0) ?   p01 :   p10;
    double   f1 = (n==0) ?   f01 :   f10;
    double phi1 = (n==0) ? phi01 : phi10;

    // simple cases
    if(phi0<=0 && phi1<=0 && phi2<=0) continue;
    if(phi0>=0 && phi1>=0 && phi2>=0) continue;

    //
    int number_of_negatives = 0;

    if(phi0<0) number_of_negatives++;
    if(phi1<0) number_of_negatives++;
    if(phi2<0) number_of_negatives++;

#ifdef CASL_THROWS
    if(number_of_negatives!=1 && number_of_negatives!=2) throw std::runtime_error("[CASL_ERROR]: Wrong configuration.");
#endif

    if(number_of_negatives==2)
    {
      phi0*=-1;
      phi1*=-1;
      phi2*=-1;
    }

    // sorting for simplication into one case
    if(phi0>0 && phi1<0) swap(phi0,phi1,f0,f1,p0,p1);
    if(phi0>0 && phi2<0) swap(phi0,phi2,f0,f2,p0,p2);
    if(phi1>0 && phi2<0) swap(phi1,phi2,f1,f2,p1,p2);

    // type : (-++)
    Point2 p_btw_01 = interpol_p(p0,phi0,p1,phi1); Point2 p_btw_02 = interpol_p(p0,phi0,p2,phi2);

    double length_of_line_segment = (p_btw_02 - p_btw_01).norm_L2();

//    sum += length_of_line_segment * 0.5*(f(p_btw_01.x,p_btw_01.y)+f(p_btw_02.x,p_btw_02.y));
    sum += length_of_line_segment * f(0.5*(p_btw_01.x+p_btw_02.x),0.5*(p_btw_01.y+p_btw_02.y));


    PetscErrorCode ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }

  return sum;
}

double Cube2::max_Over_Interface( const QuadValue& f, const QuadValue& level_set_values ) const
{
  double max = -DBL_MAX;

  Point2 p00(xyz_mmm[0],xyz_mmm[1]); double f00 = f.val[0]; double phi00 = level_set_values.val[0];
  Point2 p01(xyz_mmm[0],xyz_ppp[1]); double f01 = f.val[1]; double phi01 = level_set_values.val[1];
  Point2 p10(xyz_ppp[0],xyz_mmm[1]); double f10 = f.val[2]; double phi10 = level_set_values.val[2];
  Point2 p11(xyz_ppp[0],xyz_ppp[1]); double f11 = f.val[3]; double phi11 = level_set_values.val[3];

  // simple cases
  if(phi00<=0 && phi01<=0 && phi10<=0 && phi11<=0) return -DBL_MAX;
  if(phi00>=0 && phi01>=0 && phi10>=0 && phi11>=0) return -DBL_MAX;

  // iteration on each simplex in the Kuhn triangulation
  for(int n=0;n<2;n++)
  {
    Point2 p0=p00; double f0=f00; double phi0=phi00;
    Point2 p2=p11; double f2=f11; double phi2=phi11;

    // triangle (P0,P1,P2) with values (F0,F1,F2), (Phi0,Phi1,Phi2)
    Point2   p1 = (n==0) ?   p01 :   p10;
    double   f1 = (n==0) ?   f01 :   f10;
    double phi1 = (n==0) ? phi01 : phi10;

    // simple cases
    if(phi0<=0 && phi1<=0 && phi2<=0) continue;
    if(phi0>=0 && phi1>=0 && phi2>=0) continue;

    //
    int number_of_negatives = 0;

    if(phi0<0) number_of_negatives++;
    if(phi1<0) number_of_negatives++;
    if(phi2<0) number_of_negatives++;

#ifdef CASL_THROWS
    if(number_of_negatives!=1 && number_of_negatives!=2) throw std::runtime_error("[CASL_ERROR]: Wrong configuration.");
#endif

    if(number_of_negatives==2)
    {
      phi0*=-1;
      phi1*=-1;
      phi2*=-1;
    }

    // sorting for simplication into one case
    if(phi0>0 && phi1<0) swap(phi0,phi1,f0,f1,p0,p1);
    if(phi0>0 && phi2<0) swap(phi0,phi2,f0,f2,p0,p2);
    if(phi1>0 && phi2<0) swap(phi1,phi2,f1,f2,p1,p2);

    // type : (-++)
    double f_btw_01 = interpol_f(f0,phi0,f1,phi1); double f_btw_02 = interpol_f(f0,phi0,f2,phi2);

    max = MAX(max, MAX(f_btw_02, f_btw_01));

    PetscErrorCode ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }

  return max;
}

void Cube2::computeDistanceToInterface( const QuadValueExtended& phiAndIdxQuadValues,
		std::unordered_map<p4est_locidx_t, double>& distanceMap, const double TOL ) const
{
	// Some shortcuts.  Note the order is: x changes slowly, then y changes twice faster than x, and finally z changes
	// twice faster than y.  It's like completing a truth table.  This is the order we also followed in phiAndIdxQuadOctValues.
	const short N_POINTS = 4;
	const Point2 allPoints[N_POINTS] = {
		Point2( xyz_mmm[0], xyz_mmm[1] ),		// p00.
		Point2( xyz_mmm[0], xyz_ppp[1] ),		// p01.
		Point2( xyz_ppp[0], xyz_mmm[1] ),		// p10.
		Point2( xyz_ppp[0], xyz_ppp[1] )		// p11.
	};
	double phi00 = phiAndIdxQuadValues.val[0]; p4est_locidx_t idx00 = phiAndIdxQuadValues.indices[0];
	double phi01 = phiAndIdxQuadValues.val[1]; p4est_locidx_t idx01 = phiAndIdxQuadValues.indices[1];
	double phi10 = phiAndIdxQuadValues.val[2]; p4est_locidx_t idx10 = phiAndIdxQuadValues.indices[2];
	double phi11 = phiAndIdxQuadValues.val[3]; p4est_locidx_t idx11 = phiAndIdxQuadValues.indices[3];

	// Start with a fresh result hashmap.
	distanceMap.clear();
	distanceMap.reserve( N_POINTS );

	// If quad is not cut-out by interface there's nothing to do.
	if( phi00 <= 0 && phi01 <= 0 && phi10 <= 0 && phi11 <= 0 )
		return;
	if( phi00 > 0 && phi01 > 0 && phi10 > 0 && phi11 > 0)
		return;

	// Iterate over each simplex resulting from triangulating the quad.
	const short N_CORNERS = 3;
	for( int n = 0; n < 2; n++ )
	{
		// Defining simplex.
		const Point2* p[N_CORNERS] = { &allPoints[0], nullptr, &allPoints[3] };		// Triangle corners: still missing one of the three
		double phi[N_CORNERS] = { phi00, 0, phi11 };								// which is populated below.
		p4est_locidx_t idx[N_CORNERS] = { idx00, 0, idx11 };

		// Determine the other vertex in the triangle.
		p[1] = ( n == 0 )? &allPoints[1] : &allPoints[2];
		phi[1] = ( n == 0 )? phi01 : phi10;
		idx[1] = ( n == 0 )? idx01 : idx10;

		// Simplex not cut-out by interface: skip it.
		if( phi[0] <= 0 && phi[1] <= 0 && phi[2] <= 0 )
			continue;
		if( phi[0] > 0 && phi[1] > 0 && phi[2] > 0 )
			continue;

		// Count the number of points lying on the interface to deal with the case of an edge on the interface.
		// By convention, an exact distance of 0 is considered in the negatives side.
		std::vector<short> zeros;			// These arrays hold indices.
		std::vector<short> nonZeros;
		for( short i = 0; i < N_CORNERS; i++ )
		{
			if( ABS( phi[i] ) <= TOL )		// Is the ith point lying *on* the interface?
				zeros.push_back( i );
			else
				nonZeros.push_back( i );	// Keep track of points *not* lying on the interface.
		}

		if( zeros.size() >= 2 )
		{
			if( zeros.size() == 2 && nonZeros.size() == 1 ) 	// Validity check: there should be a single non-zero point.
				_computeDistanceToLineSegment( allPoints, phiAndIdxQuadValues, p[zeros[0]], p[zeros[1]], distanceMap, TOL );
#ifdef CASL_THROWS
			else
				throw std::runtime_error( "[CASL_ERROR]: Cube2::computeDistanceToInterface: Interface passes through all simplex points!" );
#endif
		}
		else
		{
			// Normalize to the case of -++.
			short numberOfNegatives = 0;	// Must be 1 or 2 as we have checked that not all corners have the same sign.
			for( double& i : phi )
			{
				if( i <= 0 )
				{
					numberOfNegatives++;	// Test for exact zero.  Make it slightly negative because an exact 0 causes
					if( i == 0 )			// problems with our normalization to -++.
						i = 0.0 - std::numeric_limits<double>::epsilon();
				}
			}

#ifdef CASL_THROWS
			if( numberOfNegatives != 1 && numberOfNegatives != 2 )
				throw std::runtime_error("[CASL_ERROR]: Cube2::computeDistanceToInterface: Wrong configuration!");
#endif

			if( numberOfNegatives == 2 )	// Switch signs so that we have just a single negative phi.
			{								// We perturbed exact zeros above, otherwise a case of 0, -1, +1 would give
				for( double& i : phi )		// us again 0, +1, -1, with two negatives.  This can make the cases below fail.
					i *= -1;
			}

			// Sorting for simplification into one case: -++.
			if( phi[0] > 0 && phi[1] <= 0.0) geom::utils::swapTriplet( phi[0], idx[0], p[0], phi[1], idx[1], p[1] );
			if( phi[0] > 0 && phi[2] <= 0.0) geom::utils::swapTriplet( phi[0], idx[0], p[0], phi[2], idx[2], p[2] );
			if( phi[1] > 0 && phi[2] <= 0.0) geom::utils::swapTriplet( phi[1], idx[1], p[1], phi[2], idx[2], p[2] );

			if( ABS( phi[0] ) <= TOL )		// Is p0 *on* the interface?
			{
				distanceMap[idx[0]] = 0;								// Basically, make the apex being on the interface.
				for( short i = 0; i < N_POINTS; i++ )
				{
					if( idx[0] != phiAndIdxQuadValues.indices[i] )		// Take the distance of rest of points to apex.
					{
						double d = (allPoints[i] - *p[0]).norm_L2();
						_updateMinimumDistanceMap( distanceMap, phiAndIdxQuadValues.indices[i], d );
					}
				}
			}
			else
			{
				// Obtain the line segment, L, going from the end points between p0 and p1, and between p0 and p2.
				Point2 p0_1 = geom::interpolatePoint( p[0], phi[0], p[1], phi[1], TOL );
				Point2 p0_2 = geom::interpolatePoint( p[0], phi[0], p[2], phi[2], TOL );

				// Use the above intermediate points to compute the distance from quad points to L.
				_computeDistanceToLineSegment( allPoints, phiAndIdxQuadValues, &p0_1, &p0_2, distanceMap, TOL );
			}
		}
	}
}

void Cube2::_updateMinimumDistanceMap( std::unordered_map<p4est_locidx_t, double>& distanceMap, p4est_locidx_t n, double d )
{
	distanceMap[n] = ( distanceMap.find( n ) == distanceMap.end() )? d : MIN( d, distanceMap[n] );
}

void Cube2::_computeDistanceToLineSegment( const Point2 allPoints[], const QuadValueExtended& phiAndIdxQuadValues,
										   const Point2 *v0, const Point2 *v1,
										   std::unordered_map<p4est_locidx_t, double>& distanceMap, double TOL )
{
	for( short i = 0; i < 4; i++ )
	{
		p4est_locidx_t idx = phiAndIdxQuadValues.indices[i];
		if( ABS( phiAndIdxQuadValues.val[i] ) <= TOL )		// Double check for zero distances.
			distanceMap[idx] = 0;
		else
		{
			Point2 P = geom::findClosestPointOnLineSegmentToPoint( allPoints[i], *v0, *v1, TOL );
			double d = (allPoints[i] - P).norm_L2();
			distanceMap[idx] = ( distanceMap.find( idx ) == distanceMap.end() )? d : MIN( d, distanceMap[idx] );
		}
	}
}