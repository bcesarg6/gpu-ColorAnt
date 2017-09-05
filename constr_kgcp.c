/***********************************************************
 * Created: Qua 31 Ago 2011 15:06:37 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 ***********************************************************
 *
 * A simple constructive algorithm for the kpcg:
 * * try to find the least color for a vertex;
 * * if there is no such color, choose a random one;
 * 	
 ***********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "color.h"
#include "util.h"
#include "constructive.h"

void constr_kgcp_printbanner(void) {
  fprintf(problem->fileout, "SIMPLE CONSTRUCTIVE ALGORITHM FOR K-PCG\n");
  fprintf(problem->fileout, "-------------------------------------------------\n");
  fprintf(problem->fileout, "Parameters:\n");
}

void constr_kgcp_initialization(void) {
  return;
}

void constr_kgcp_malloc(void) {
  return;
}

void constr_kgcp_show_solution(void) {
  return;
}

gcp_solution_t* constr_kgcp(void) {

  gcp_solution_t *solution = init_solution();
  int nc, i, c, max_degree, color, v, j;
  int possible_color[problem->nof_vertices];
  int neighbors_by_color[problem->nof_vertices][problem->max_colors+1];
  int confl_vertices[problem->nof_vertices];

  solution->nof_colors = problem->max_colors;

  /* Initializing auxiliary arrays and choosing a vertex with a maximal degree */
  max_degree = 0;
  for (i = 0; i < problem->nof_vertices; i++) {
    possible_color[i] = 0;
    confl_vertices[i] = 0;
    solution->color_of[i] = -1;
    for (j = 0; j <= problem->max_colors; j++) {
      neighbors_by_color[i][j] = 0;
    }
    if (problem->degree[i] > problem->degree[max_degree]) {
      max_degree = i;
    }
  }

  /* Color the chosen vertex with the first color (0) */
  color = 0;
  solution->color_of[max_degree] = color;
  v = max_degree;	/* the actual vertex, last one that was colored */
  nc = 1;			/* number of colored vertices */

  solution->nof_confl_edges = 0;
  solution->nof_confl_vertices = 0;

  while (nc < problem->nof_vertices) {

    max_degree = -1;
    for (i = 0; i < problem->nof_vertices; i++) {
      if (problem->adj_matrix[v][i]) {
	/* update degree of saturation: */
	if (neighbors_by_color[i][color] == 0) {
	  neighbors_by_color[i][problem->max_colors]++;
	}
	/* now <i> has a neighbor colored with <color> */
	neighbors_by_color[i][color]++;

	if (solution->color_of[i] == -1) {
	  /* if <i> is not colored yet and <i> is neighbor of <v>,
	   * update possible color for <i>: among all the possible 
	   * colors for a neighbor of <i>, chose the least one */
	  int changed = FALSE;
	  for (c = problem->max_colors; c >= 0; c--) {
	    if (neighbors_by_color[i][c] == 0) {
	      possible_color[i] = c;
	      changed = TRUE;
	    }
	  }
	  if (!changed) possible_color[i] = problem->max_colors;
	}
      }
      if (solution->color_of[i] == -1) {
	//	/* Find the vertex with maximal degree between the 
	//	 * non-colored ones */
	//	if (max_degree == -1) max_degree = i;
	//	if (problem->degree[i] > problem->degree[max_degree]) {
	//		max_degree = i;
	//	}
	if (max_degree == -1) max_degree = i;
	else if (neighbors_by_color[i][problem->max_colors] >
		 neighbors_by_color[max_degree][problem->max_colors]) {
	  max_degree = i;
	}			
      }
    }

    v = max_degree;
    color = possible_color[v];

    /* if no viable colors is found for <v>, chose a random one.
     * this means that a conflict is being generated. */
    if (color == problem->max_colors) {
#if defined LRAND
      //color = (int) RANDOM(problem->max_colors);
      RANDOM(problem->buffer, color, int, problem->max_colors);
#elif defined NRAND
      //color = (int) RANDOM(problem->max_colors);
      RANDOM(problem->seed, problem->buffer, color, int, problem->max_colors);
#endif

      int checked = 0;
      int conf = solution->nof_confl_edges;
      for (i = 0; i < problem->nof_vertices; i++) {
	if (problem->adj_matrix[i][v] && solution->color_of[i] == color) {
	  solution->nof_confl_edges++;
	  checked++;
	  if (confl_vertices[i] == 0) {
	    confl_vertices[i] = 1;
	    solution->nof_confl_vertices++;
	  }					
	}
      }
      if (conf != solution->nof_confl_edges) {
	checked++;
	if (confl_vertices[v] == 0) {
	  confl_vertices[v] = 1;
	  solution->nof_confl_vertices++;
	}
      }
      if (checked == problem->nof_vertices) break;

    }

    solution->color_of[v] = color;
    nc++;

  }

  solution->spent_time = current_time_secs(TIME_FINAL, time_initial);
  solution->time_to_best = solution->spent_time;

  return solution;

}


