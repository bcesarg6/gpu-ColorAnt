/*************************************************************************
 * Created: Ter 01 Fev 2011 22:50:55 BRST
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 *************************************************************************/
#include <stdlib.h>
#include <sys/resource.h>
#include "util.h"

double time_initial, time_final;

static double current_gettime_secs(void) {
  struct timeval time;

  gettimeofday(&time, NULL);

  return (time.tv_sec + (time.tv_usec/1000000.0));
}


static double current_usertime_secs(void) {/*{{{*/
    double usertime;
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) < 0) {
        perror("get rusage");
        return -1;
    }

    usertime = usage.ru_utime.tv_sec + (usage.ru_utime.tv_usec * 1e-6);

    return usertime;
}/*}}}*/

static double current_alltime_secs(void) {/*{{{*/
    double usertime, systemtime;
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) < 0) {
        perror("get rusage");
        return -1;
    }

    usertime = usage.ru_utime.tv_sec + (usage.ru_utime.tv_usec * 1e-6);
    systemtime = usage.ru_stime.tv_sec + (usage.ru_stime.tv_usec * 1e-6);

    return (usertime + systemtime);
}/*}}}*/

double current_time_secs(int flag, double initial) {

 switch(flag){
   case TIME_INITIAL:
        return current_gettime_secs();
        break;
   case TIME_FINAL:
        return current_gettime_secs() - initial;
        break;
   case TIME_USERTIME:
        return current_usertime_secs();
        break;
   case TIME_ALLTIME:
        return current_alltime_secs();
        break;
 }

 return 0;

}

void* malloc_(size_t size) {/*{{{*/
    void *p;
        p = malloc(size); //Em CUDA mallocs precisam de cast! (size_t*) adicionado, se der erro tirar!

    if (!p) {
        perror("malloc");
        abort();
    }
    return p;
}/*}}}*/

long int create_seed(void) {/*{{{*/
    long int gseed;
    struct timeval tval;

    if (gettimeofday(&tval, NULL) < 0) {
        perror("get time of day");
        return 1;
    }

    gseed = tval.tv_sec * tval.tv_usec;

    return gseed;
}/*}}}*/

#if defined NRAND
unsigned long int print_seed(unsigned short *seed) {
        unsigned long int ret;
        memcpy(&ret, seed, sizeof(unsigned short)*3);
        return ret;
}
#endif
