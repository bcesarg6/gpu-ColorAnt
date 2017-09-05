/***********************************************************
 * Created: Ter 09 Ago 2011 21:05:59 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 ***********************************************************
 *
 * k-ANTCOL [1997, Ants can colour graphs, Costa and Hertz]
 * * Using ANT_FIXES_K as constructive method;
 *
 ***********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "color.h"
#include "util.h"
#include "aco.h"
#include "ant_fixed_k.h"

#define OUT -1

/* Global data */
static double **pheromone;
static double **phero_var;
static gcp_solution_t *ant_k;
static gcp_solution_t *best_ant;

void kantcol_printbanner(void) {
  fprintf(problem->fileout, "KANTCOL\n");
  fprintf(problem->fileout, "-------------------------------------------------\n");
  fprintf(problem->fileout, "Parameters:\n");
 if (!(get_flag(problem->flags, FLAG_RATIO))) { 
	fprintf(problem->fileout, "  Ants.............................: %i\n", aco_info->nants);
  }
  else {
	fprintf(problem->fileout, "  Ants.............................: %i (%i of %i - vertices)\n", aco_info->nants, aco_info->ratio, problem->nof_vertices);
  }
  fprintf(problem->fileout, "  Alpha............................: %.2f\n", aco_info->alpha);
  fprintf(problem->fileout, "  Beta.............................: %.2f\n", aco_info->beta);
  fprintf(problem->fileout, "  Rho..............................: %.2f\n", aco_info->rho);
}

void kantcol_malloc(void) {
	
  aco_info = malloc_(sizeof(aco_t));
  aco_info->nants = KANTCOL_NANTS;
  aco_info->ratio = KANTCOL_NANTS;
  aco_info->alpha = KANTCOL_ALPHA;
  aco_info->beta = KANTCOL_BETA;
  aco_info->rho = KANTCOL_RHO;

}

void kantcol_initialization(void) {

  if (!(get_flag(problem->flags, FLAG_COLOR))) {
    problem->max_colors = problem->nof_vertices;
  }

  if (get_flag(problem->flags, FLAG_RATIO)) {
        aco_info->ratio = aco_info->nants;
	aco_info->nants = (problem->nof_vertices * aco_info->nants) / 100;
  }

}

void kantcol_show_solution(void) {
  return;
}

/* Functions to help updating pheromone */

static void update_var_phero(gcp_solution_t *solution) {

  int i, j;

  for (i = 0; i < problem->nof_vertices; i++) {
    for (j = 0; j < problem->nof_vertices; j++) {
      if (!problem->adj_matrix[i][j] && 
	  (solution->color_of[i] == solution->color_of[j])) {
	phero_var[i][j] += (solution->nof_confl_vertices == 0) ?
	  1 : 1.0/solution->nof_confl_vertices;
      }
    }
  }

}

static void update_pheromone_trails(void) {

  int i, j;

  for (i = 0; i < problem->nof_vertices; i++) {
    for (j = 0; j < problem->nof_vertices; j++) {
      if (!problem->adj_matrix[i][j]) {
	pheromone[i][j] *= aco_info->rho;
	pheromone[i][j] += phero_var[i][j];
      }

      phero_var[i][j] = 0;
    }
  }

}

static void initialize_data(void) {
  int i, j;

  pheromone = malloc_(sizeof(double*) * problem->nof_vertices);
  phero_var = malloc_(sizeof(double*) * problem->nof_vertices);

  for (i = 0; i < problem->nof_vertices; i++) {
    pheromone[i] = malloc(sizeof(double) * problem->nof_vertices);
    phero_var[i] = malloc(sizeof(double) * problem->nof_vertices);
    for (j = 0; j < problem->nof_vertices; j++) {
      pheromone[i][j] = 0;
      phero_var[i][j] = 0;
      if (problem->adj_matrix[i][j] == 0) {
	pheromone[i][j] = 1;
      }
    }
  }

  best_ant = malloc_(sizeof(gcp_solution_t));
  best_ant->color_of = malloc_(sizeof(int) * problem->nof_vertices);
  best_ant->nof_confl_vertices = INT_MAX;	
  best_ant->nof_colors = problem->max_colors;

  ant_k = malloc_(sizeof(gcp_solution_t));
  ant_k->color_of = malloc_(sizeof(int) * problem->nof_vertices);
  ant_k->nof_confl_vertices = INT_MAX;	
  ant_k->nof_colors = problem->max_colors;

  afk_initialize_data(aco_info->alpha, aco_info->beta);

}


static int construct_solutions(int cycle, int converg) {

  int k;
  for (k = 0; k < aco_info->nants; k++) {
    ant_fixed_k(ant_k, pheromone);
    ant_k->total_cycles = cycle;
    ant_k->spent_time = current_time_secs(TIME_FINAL, time_initial);
    if (ant_k->nof_confl_vertices < best_ant->nof_confl_vertices) {
      cpy_solution(ant_k, best_ant);
      best_ant->cycles_to_best = cycle;
      best_ant->time_to_best = ant_k->spent_time;
      converg = 0;
    }

    update_var_phero(ant_k);
  }
  return converg;
}

gcp_solution_t* kantcol(void) {

  int cycle = 0;
  int converg = 0;

  initialize_data();
  best_ant->stop_criterion = 0;	

  while (!terminate_conditions(best_ant, cycle, converg)) {

    cycle++;
    converg++;

    converg = construct_solutions(cycle, converg);
    update_pheromone_trails();
   

    if (get_flag(problem->flags, FLAG_VERBOSE)) {
  fprintf(problem->fileout, "Cycle %d - Conflicts found: %d (edges), %d (vertices)\n", cycle, best_ant->nof_confl_edges, best_ant->nof_confl_vertices);

    }

    if (best_ant->nof_confl_vertices == 0) {
      best_ant->stop_criterion = STOP_BEST;
      break;
    }
  }

  best_ant->spent_time = current_time_secs(TIME_FINAL, time_initial);
  best_ant->total_cycles = cycle;	

  return best_ant;

}

