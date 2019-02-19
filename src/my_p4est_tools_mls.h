#ifndef MY_P4EST_MLS_TOOLS_H
#define MY_P4EST_MLS_TOOLS_H

#include <vector>
#include <src/my_p4est_integration_mls.h>

#ifdef P4_TO_P8
class level_set_tot_t : public CF_3
#else
class level_set_tot_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;
  std::vector<bool>     *everywhere;

public:

#ifdef P4_TO_P8
  level_set_tot_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, std::vector<bool> *everywhere = NULL) :
#else
  level_set_tot_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, std::vector<bool> *everywhere = NULL) :
#endif
    phi_cf(phi_cf), action(action), color(color), everywhere(everywhere) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (unsigned short i = 0; i < phi_cf->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current < phi_total) phi_total = phi_current;
      }
    }

    if (everywhere != NULL)
    {
      for (unsigned short i = 0; i < phi_cf->size(); ++i)
      {
        if (everywhere->at(i) && action->at(i) != COLORATION)
#ifdef P4_TO_P8
          phi_total = MIN(phi_total, fabs((*phi_cf->at(i))(x,y,z)));
#else
          phi_total = MIN(phi_total, fabs((*phi_cf->at(i))(x,y)));
#endif
      }
    }

    return phi_total;
  }
};

#ifdef P4_TO_P8
class level_set_smooth_t : public CF_3
#else
class level_set_smooth_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;
  double epsilon;

public:

#ifdef P4_TO_P8
  level_set_smooth_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, double epsilon) :
#else
  level_set_smooth_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color, double epsilon) :
#endif
    phi_cf(phi_cf), action(action), color(color), epsilon(epsilon) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (unsigned short i = 0; i < phi_cf->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        phi_total = 0.5*(phi_total+phi_current+sqrt(SQR(phi_total-phi_current)+epsilon));
//        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        phi_total = 0.5*(phi_total+phi_current-(sqrt(SQR(phi_total-phi_current)+epsilon)-(epsilon)/sqrt(SQR(phi_total-phi_current)+epsilon)));
//        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
    return phi_total;//+epsilon;
  }
};

#endif // MY_P4EST_MLS_TOOLS_H
