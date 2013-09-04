#include "CASL_math.h"

double MINMOD(double a, double b)
{
    if(a*b<=0) return 0;
    else
    {
        if((fabs(a))<(fabs(b))) return a;
        else                  return b;
    }
}


double SIGN(double a)
{
    return (a>0) ? 1:-1;
}

double interface_Location( double   a, double   b, double  fa, double  fb )
{
    return 0.5*(a+b+(a-b)*(fa+fb)/(fb-fa));
}

template <class  T>
double CBRT(const T& x){
    return     ((x) > 0.0 ?  pow((double) (x), 1.0/3.0) : \
                             ((x) < 0.0 ? -pow((double)-(x), 1.0/3.0) : 0.0));
}

double interface_Location_With_First_Order_Derivative(	double   a, double   b,
                                                        double  fa, double  fb,
                                                        double fxa, double fxb )
{
#ifdef CASL_THROWS
    if(fa*fb >= 0) throw std::invalid_argument("[CASL_ERROR]: Wrong arguments.");
#endif

    // third order polynomial
    // f(x) = c0 + c1*(x-a) + c2*(x-a)^2 + c3*(x-a)^3
    double  h = b-a; double c[4],s[3];
    c[0] = fa;
    c[1] = fxa;
    c[2] = (3*(fb-fa)-h*(2*fxa+fxb))/(h*h);
    c[3] = (2*(fa-fb)+h*(fxa+fxb))/(h*h*h);

    int number_of_roots = solve_Cubic(c,s);

    double s_valid[3]; int number_of_valid_solution=0;

    // valid solution is in [0,h]
    for(int i=0;i<number_of_roots;i++)
    {
        if(s[i]>=0 && s[i]<=h){ s_valid[number_of_valid_solution++]=s[i]; }
    }

#ifdef CASL_THROWS
    if(number_of_valid_solution < 1) { printf("ouiiiiiii %f %f %f %f %f %f %d\n",a, b, fa, fb, fxa, fxb, b>0); throw std::invalid_argument("[CASL_ERROR]: Wrong arguments."); }
#endif

    if(number_of_valid_solution==1) return s_valid[0]+a;
    else                            return(s_valid[0]+s_valid[1])/2+a;
}

int solve_Linear( double c[2], double s[1] )
{
    if(is_Zero(c[1]))                     return 0;
    else            { s[0] = -c[0]/c[1]; return 1; }
}

int solve_Quadric( double c[3], double s[ 2 ])
{
    if(is_Zero(c[2])) return solve_Linear(c,s);
    else
    {
        /* normal form: x^2 + 2*px + q = 0 */
        double p = c[ 1 ] / (2 * c[ 2 ]);
        double q = c[ 0 ] / c[ 2 ];
        double D = p*p - q;
        if(is_Zero(D))
        {
            s[0] = - p;return 1;
        }
        else
            if(   D < 0 )
            {
                return 0;
            }
            else
            {
                double sqrt_D = sqrt(D);
                s[ 0 ] =   sqrt_D - p;
                s[ 1 ] = - sqrt_D - p;
                return 2;
            }
    }
}
int solve_Cubic(double c[ 4 ], double s[ 3 ])
{
    if(is_Zero(c[3])) return solve_Quadric(c,s);
    else
    {
        /* normal form: x^3 + Ax^2 + Bx + C = 0 */
        double A = c[ 2 ] / c[ 3 ];
        double B = c[ 1 ] / c[ 3 ];
        double C = c[ 0 ] / c[ 3 ];

        /*  substitute x = y - A/3 to eliminate quadric term:
            x^3 +px + q = 0 */
        double sq_A = A * A;
        double p = 1.0/3 * (- 1.0/3 * sq_A + B);
        double q = 1.0/2 * (2.0/27 * A * sq_A - 1.0/3 * A * B + C);


        /* use Cardano's formula */
        double cb_p = p * p * p;
        double D = q * q + cb_p;
        int num;
        if(is_Zero(D))
        {
            if (is_Zero(q)) 	/* one triple solution */
            {
                s[ 0 ] = 0;
                num = 1;
            }
            else 			/* one single and one double solution */
            {
                double u = CBRT(-q);
                s[ 0 ] = 2 * u;
                s[ 1 ] = - u;
                num = 2;
            }
        }
        else if (D < 0) /* Casus irreducibilis: three real solutions */
        {
            double phi = 1.0/3 * acos(-q / sqrt(-cb_p));
            double t = 2 * sqrt(-p);

            s[ 0 ] =   t * cos(phi);
            s[ 1 ] = - t * cos(phi + PI / 3);
            s[ 2 ] = - t * cos(phi - PI / 3);
            num = 3;
        }
        else 			/* one real solution */
        {
            double sqrt_D = sqrt(D);
            double u =   CBRT(sqrt_D - q);
            double v = - CBRT(sqrt_D + q);

            s[ 0 ] = u + v;
            num = 1;
        }

        /* resubstitute */
        double sub = 1.0/3 * A;

        for(int i = 0; i < num; ++i)
            s[ i ] -= sub;

        return num;
    }
}

int is_Zero(double x)
{
    return x > - EPSILON && x < EPSILON;
}
