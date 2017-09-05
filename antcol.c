/***********************************************************
 * Created: Ter 09 Ago 2011 21:05:59 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 ***********************************************************
 *
 * ANTCOL [1997, Ants can colour graphs, Costa and Hertz]
 * * Using RLF as constructive method;
 * * First vertex of a class is chosen randomly;
 * * Other vertices chosen using heuristic information = deg_B(v)
 * * alpha = 2; beta = 4; rho = 0.5; ncycles = 50; nants = {100,300}
 *
 ***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "color.h"
#include "util.h"
#include "aco.h"

#define OUT -1

/* Global data */
static double **pheromone;
static double **phero_var;
static gcp_solution_t *ant_k;
static gcp_solution_t *best_ant;

/* ANTCOL data */
static double *probb;
static double **trail;
static int *degree;
static int **adj;

void antcol_printbanner(void) {
  fprintf(problem->fileout, "ANTCOL\n");
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

void antcol_malloc(void) {
	
  aco_info = malloc_(sizeof(aco_t));
  aco_info->nants = ANTCOL_NANTS;
  aco_info->ratio = ANTCOL_NANTS;
  aco_info->alpha = ANTCOL_ALPHA;
  aco_info->beta = ANTCOL_BETA;
  aco_info->rho = ANTCOL_RHO;

}

void antcol_initialization(void) {

  problem->max_colors = problem->nof_vertices;

  if (get_flag(problem->flags, FLAG_RATIO)) {
        aco_info->ratio = aco_info->nants;
	aco_info->nants = (problem->nof_vertices * aco_info->nants) / 100;
  }

}

void antcol_show_solution(void) {
  return;
}


/* Functions to help ANT_RLF */

static int first_vertex(void) {
  int v;
#if defined LRAND
  //v = RANDOM(problem->nof_vertices);
  RANDOM(problem->buffer, v, int, problem->nof_vertices);
#elif defined NRAND
  //v = RANDOM(problem->seed, problem->nof_vertices);
  RANDOM(problem->seed, problem->buffer, v, int, problem->nof_vertices);
#endif

  while (degree[v] == OUT) {
#if defined LRAND
    //v = RANDOM(problem->nof_vertices);
    RANDOM(problem->buffer, v, int, problem->nof_vertices);
#elif defined NRAND
    //v = RANDOM(problem->seed, problem->nof_vertices);
    RANDOM(problem->seed, problem->buffer, v, int, problem->nof_vertices);
#endif
  }

  return v;
}

static void calculate_probbs(int v, int color, int *color_of, int *size_color) {
	
  int i, j, neighbors;
  double sum, traill, totalsum;

  totalsum = 0;

  for (i = 0; i < problem->nof_vertices; i++) {
		
    probb[i] = 0;
		
    /* one have to choose a vertex that belongs to the set of vertices not
     * colored yet (those ones whose <degree != OUT>) and still viable to 
     * be colored with <color> (those ones that are not adjacent to <v> 
     * and <v> itself) */
    if ((degree[i] != OUT) && (i!=v) && (!adj[i][v])) {
      sum = 0;
      neighbors = 1;

      for (j = 0; j < problem->nof_vertices; j++) {
	if (color_of[j] == color) {
	  sum += pheromone[j][i];
	}
	/* the heuristic information about a vertex <i> is the number
	 * of neighbors of <i> that belong to the set vertices not 
	 * colored yet (those ones whose <degree != OUT>) that are 
	 * not viable to be colored with <color> (those ones that are
	 * adjacent to <v>) */
	if ((degree[j] != OUT) && adj[j][v] == 1 && adj[j][i] == 1) {
	  neighbors++;
	}
      }

      traill = (size_color[color] == 0) ? 1 : sum/size_color[color];

      probb[i] = pow(traill, aco_info->alpha) * pow(neighbors, aco_info->beta);
      totalsum += probb[i];
    }
  }

  probb[problem->nof_vertices] = totalsum;

}

static int choose_vertex(void) {
  int i;
  double p, last, div;

  div = probb[problem->nof_vertices];

#if defined LRAND
  //p = (double) RANDOM_UNIT() / INT_MAX;
  RANDOM_UNIT(problem->buffer, p, double);
  p = p / INT_MAX;
#elif defined NRAND
  //p = (double) RANDOM_UNIT(problem->seed) / INT_MAX;
  RANDOM_UNIT(problem->seed, problem->buffer, p, double);
  p = p / INT_MAX;
#endif

  last = 0;
  for (i = 0; i < problem->nof_vertices; i++) {
    last += (probb[i]/div);
    if (p <= last) {
      return i;
    }
  }
  return problem->nof_vertices-1;	
}

static void contract(int v1, int v2, int *nn) {
  /* Constracts <v1> and <v2> into one vertex, represented by <v1> */	
  int i;
  for (i = 0; i < problem->nof_vertices; i++) {
    if (adj[i][v2]) {
			
      degree[i]--;
      adj[i][v2] = adj[v2][i] = FALSE;

      if (!adj[i][v1] && (degree[i] != OUT)) {
	adj[i][v1] = adj[v1][i] = TRUE;
	degree[v1]++;
	degree[i]++;
	(*nn)--;
      }

    }
  }
  degree[v2] = OUT;
}

static void remove_(int v) {
  /* Removes <v> from the graph */	
  int i;
  for (i = 0; i < problem->nof_vertices; i++) {
    if (adj[v][i]) {
      adj[v][i] = adj[i][v] = FALSE;
      degree[i]--;
    }
  }
  degree[v] = OUT;
}

static void ant_rlf(gcp_solution_t *sol) {

  int i, j;
  int colornumber = 0;	/* number of colors used */
  int colored = 0;		/* number of colored vertex */
  int v, new_v;			/* vertex to be colored */
  int nn;					/* keep number of non-neighbors of v */
	
  int size_color[problem->max_colors];
	
  for (i = 0; i < problem->nof_vertices; i++) {
    degree[i] = problem->degree[i];
    sol->color_of[i] = OUT;
    size_color[i] = 0;
    for (j = 0; j < problem->nof_vertices; j++) {
      adj[i][j] = problem->adj_matrix[i][j];
    }
  }

  while (colored < problem->nof_vertices) {

    v = first_vertex();
    sol->color_of[v] = colornumber;
    size_color[colornumber] = 1;
    colored++;

    /* calculate number of non-neighbors of <v> viable to be
     * colored with <colornumber> */		
    nn = problem->nof_vertices - degree[v] - 1;
    for (i = 0; i < problem->nof_vertices; i++) {
      if (!adj[v][i] && (degree[i] == OUT)) {
	nn--;
      }
    }

    while (nn > 0) {
      /* choose new_v, non-neighbor of v, that is not OUT with probability p */
      calculate_probbs(v, colornumber, sol->color_of, size_color);
      new_v = choose_vertex();

      sol->color_of[new_v] = colornumber;
      size_color[colornumber]++;
      colored++;

      contract(v, new_v, &nn);

      nn--;
    }

    remove_(v);
    colornumber++;

  }

  sol->spent_time = current_time_secs(TIME_FINAL, time_initial);
  sol->nof_colors = colornumber;

}

static void update_var_phero(gcp_solution_t *solution) {
  int i, j;
  for (i = 0; i < problem->nof_vertices; i++) {
    for (j = i; j < problem->nof_vertices; j++) {
      if (problem->adj_matrix[i][j] == 0 && (solution->color_of[i] == solution->color_of[j])) {
	phero_var[i][j] += 1.0 / solution->nof_colors;
	phero_var[j][i] = phero_var[i][j];
      }
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
      if (problem->adj_matrix[i][j] == 0)
	pheromone[i][j] = 1;
    }
  }

  best_ant = malloc_(sizeof(gcp_solution_t));
  best_ant->color_of = malloc_(sizeof(int) * problem->nof_vertices);
  best_ant->nof_colors = INT_MAX;

  ant_k = malloc_(sizeof(gcp_solution_t));
  ant_k->color_of = malloc_(sizeof(int) * problem->nof_vertices);

  probb = malloc_(sizeof(double) * (problem->nof_vertices+1));	
  degree = malloc_(sizeof(int) * problem->nof_vertices);
  adj = malloc_(sizeof(int*) * problem->nof_vertices);
  trail = malloc_(sizeof(double*) * problem->max_colors);

  for (i = 0; i < problem->nof_vertices; i++) {
    adj[i] = malloc(sizeof(int) * problem->nof_vertices);
    trail[i] = malloc_(sizeof(double) * problem->nof_vertices);
    for (j = 0; j < problem->nof_vertices; j++) {
      trail[i][j] = 0;
    }
  }

}

static void initialize_var_phero(void) {
  int i, j;
  for (i = 0; i < problem->nof_vertices; i++) {
    for (j = i; j < problem->nof_vertices; j++) {
      if (problem->adj_matrix[i][j] == 0) {
	phero_var[i][j] = phero_var[j][i] = 0;
      }
    }
  }
}

static int construct_solutions(int cycle, int converg) {
  int k;
  for (k = 0; k < aco_info->nants; k++) {

    ant_rlf(ant_k);
    ant_k->spent_time = current_time_secs(TIME_FINAL, time_initial);

    if (ant_k->nof_colors < best_ant->nof_colors) {
      cpy_solution(ant_k, best_ant);
      best_ant->cycles_to_best = cycle;
      best_ant->time_to_best = ant_k->spent_time;
      converg = 0;
    }

    update_var_phero(ant_k);
  }
  return converg;
}

static void update_pheromone_trails(void) {
  int i, j;
  for (i = 0; i < problem->nof_vertices; i++) {
    for (j = i; j < problem->nof_vertices; j++) {
      if (!problem->adj_matrix[i][j]) {
	pheromone[i][j] = (aco_info->rho * pheromone[i][j]) + phero_var[i][j];
	pheromone[j][i] = pheromone[i][j];
      }
    }
  }
}

gcp_solution_t* antcol(void) {

  int cycle = 0;
  int converg = 0;

  initialize_data();

  while (!terminate_conditions(best_ant, cycle, converg)) {

    cycle++;
    converg++;

    initialize_var_phero();
    converg = construct_solutions(cycle, converg);
    update_pheromone_trails();

    if (get_flag(problem->flags, FLAG_VERBOSE)) {
      printf("Cycle %d - No. of colors used: %d\n", cycle, best_ant->nof_colors);
    }

  }

  best_ant->spent_time = current_time_secs(TIME_FINAL, time_initial);
  best_ant->total_cycles = cycle;	

  return best_ant;

}

