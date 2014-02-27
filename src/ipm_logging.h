#ifndef _IPM_LOGGING_H_
#define _IPM_LOGGING_H_

#ifdef IPM_LOG_EVENTS
#define IPMLogRegionBegin(e) MPI_Pcontrol(1, e)
#define IPMLogRegionEnd(e) MPI_Pcontrol(-1, e)
#define IPMLogEvent(e) MPI_Pcontrol(0, e)
#else
#define IPMLogRegionBegin(e) (void)0
#define IPMLogRegionEnd(e) (void)0
#define IPMLogEvent(e) (void)0
#endif // IPM_LOG_EVENTS

#endif // _IPM_LOGGING_H_
