/*************************************************************************
 * Created: Ter 01 Fev 2011 22:59:46 BRST
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 *************************************************************************/




#ifndef __UTIL_H
#define __UTIL_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "color.h"

/* times */
#define TIME_INITIAL    1
#define TIME_FINAL      2
#define TIME_USERTIME   4
#define TIME_ALLTIME    8

#if defined LRAND

#define RANDOM(buffer, result, cast, limit) {long int value; lrand48_r(&buffer, &value); result = (cast) (value % limit);}
#define RANDOM_UNIT(buffer, result, cast)   {long int value; lrand48_r(&buffer, &value); result = (cast) value;}

#elif defined NRAND

#define RANDOM(seed, buffer, result, cast, limit) {long int value; nrand48_r(seed, &buffer, &value); result = (cast) (value % limit);}
#define RANDOM_UNIT(seed, buffer, result, cast)   {long int value; nrand48_r(seed, &buffer, &value); result = (cast) value;}

#endif

#define bool int
#define TRUE 1
#define FALSE 0

extern double time_initial, time_final;

void* malloc_(size_t size);
double current_time_secs(int, double);
long int create_seed(void);

#if defined NRAND

unsigned long int print_seed(unsigned short *seed);

#endif

#endif /* __UTIL_H */
