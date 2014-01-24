#ifndef _IPM_LOGGING_H_
#define _IPM_LOGGING_H_

#ifdef IPM_LOG_EVENTS
#define IPMLogEventBegin(e) MPI_Pcontrol(1, e)
#define IPMLogEventEnd(e) MPI_Pcontrol(-1, e)
#else
#define IPMLogEventBegin(e) 0
#define IPMLogEventEnd(e) 0
#endif // IPM_LOG_EVENTS

#endif // _IPM_LOGGING_H_
