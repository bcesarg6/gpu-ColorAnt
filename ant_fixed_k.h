/**********************************************************
 * Created: Seg 10 Out 2011 17:23:49 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 * ANT_FIXED_K
 * * Constructive method for each ant in an ACO algorithm for k-GCP
 *
 **********************************************************/

#ifndef __ANT_FIXED_K_H
#define __ANT_FIXED_K_H

void afk_initialize_data(float p_alpha, float p_beta);
void ant_fixed_k(gcp_solution_t *solution, double **pheromone);

#endif /* __ANT_FIXED_K_H */
