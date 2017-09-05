/**********************************************************
 * Created: Dom 04 Set 2011 23:38:53 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 **********************************************************
 *
 * Simple constructive algorithm for the kpcg:
 * * try to find the least color for a vertex;
 * * if there is no such color, choose a random one;
 * 	
 ***********************************************************/
#ifndef __CONSTRUCTIVE_H
#define __CONSTRUCTIVE_H

#if defined CONSTRKGCP || defined TABUCOL

void constr_kgcp_printbanner(void);
void constr_kgcp_initialization(void);
void constr_kgcp_malloc(void);
void constr_kgcp_show_solution(void);
gcp_solution_t* constr_kgcp(void);

#endif
 
#endif /* __CONSTRUCTIVE_H */

