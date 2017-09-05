/***********************************************************
 * Created: Seg 08 Ago 2011 17:01:23 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 ***********************************************************
 *
 * Local search Tabucol [1987, Hertz]
 * Works with infeasible k-colorings.
 * Can be used with either a react or a dynamic tabu tenure.
 *
 * This program is an adaptation of the code available in
 * http://rose.epfl.ch/~bloechli/coloring/
 * explained below.
 *
 ***********************************************************/
/******************************************************************************/
//
//  ReactPartialCol, PartialCol, ReactTabucol and Tabucol graph coloring
//  heuristics. Reference code for the paper
//  "A Reactive Tabu Search Using Partial Solutions for the
//  Graph Coloring Problem" by Ivo Bloechliger and Nicolas Zuffery.
//
//  Copyright (C) 2003 - Ecole Poyltechnique Federale de Lausanne - EPFL,
//  Laboratory of Operations Research South Est-ROSE, 1015 Lausanne, Switzerland
//  Written by Ivo Bloechliger, Ivo.Bloechliger@epfl.ch
//  http://rose.epfl.ch/~bloechli/coloring/
//
/******************************************************************************/
//
// This program is distributed under the terms of the GNU General Public License
// as published by the Free Software Foundation. In paticular, this program is
// distributed WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. The EPFL shall in no
// case be liable for any damage of any kind in connection with the use of this
// program.  See the GNU General Public License for more details
// (http://www.gnu.org/copyleft/gpl.html#SEC1).
//
/******************************************************************************/

#include <stdio.h>
#include "color.h"
#include "tabucol.h"
#include "util.h"

void tabucol_printbanner(void) {/*{{{*/

  fprintf(problem->fileout, "TABUCOL\n");
  fprintf(problem->fileout, "-------------------------------------------------\n");
  if (tabucol_info->cycles > 0) {
    fprintf(problem->fileout, "Parameters:\n");
    fprintf(problem->fileout, "  Iterations.............................: %i\n", tabucol_info->cycles);
    if (get_flag(problem->flags, FLAG_TABUCOL_CONV)) fprintf(problem->fileout, "  Stop tabucol after %i cycles without improvement.\n", tabucol_info->convergence_cycles);

    if (get_flag(problem->flags, FLAG_CHANGE_TABUCOL_SCHEME)) {
      fprintf(problem->fileout, "  Change tabucol scheme after %d cycles.\n", tabucol_info->change_scheme_iterations);
      fprintf(problem->fileout, "\tInitial scheme: %s\n", tabucol_info->tl_style == TABUCOL_REACTIVE ? "Reactive" : "Dynamic");
    }

    if (get_flag(problem->flags, FLAG_DIFF_TABUCOL_SCHEME)) {
      fprintf(problem->fileout, "  Different tabucol scheme after %d cycles.\n", tabucol_info->change_scheme_iterations);
      fprintf(problem->fileout, "\tInitial scheme: %s\n", tabucol_info->tl_style == TABUCOL_REACTIVE ? "Reactive" : "Dynamic");
    }

    if ( (!(get_flag(problem->flags, FLAG_CHANGE_TABUCOL_SCHEME))) && (!(get_flag(problem->flags, FLAG_DIFF_TABUCOL_SCHEME))) ){
      if (tabucol_info->tl_style == TABUCOL_REACTIVE)
	fprintf(problem->fileout, "  Reactive scheme for tabu tenure.\n");
      else
	fprintf(problem->fileout, "  Dynamic scheme for tabu tenure.\n");
    }

    if (get_flag(problem->flags, FLAG_TABUCOL_ALL_ANTS))
      fprintf(problem->fileout, "  Apply tabu search on all ants.\n");
    else
      fprintf(problem->fileout, "  Apply tabu search only on the best ant.\n");

  }
  else
    fprintf(problem->fileout, "  Do not use tabu search.\n");

}/*}}}*/

void tabucol_initialization(void) {

}

void tabucol_malloc(void) {/*{{{*/

  tabucol_info = (tabucol_t*) malloc_(sizeof(tabucol_t));
  tabucol_info->tl_style = TABUCOL_DYNAMIC;
  tabucol_info->cycles = TABUCOL_CYCLES;
  tabucol_info->convergence_cycles = TABUCOL_CONVERGENCE_CYCLES;
  tabucol_info->change_scheme_iterations = TABUCOL_CHANGE_SCHEME_ITERATIONS;
  tabucol_info->diff_scheme_iterations = TABUCOL_DIFF_SCHEME_ITERATIONS;
  tabucol_info->spent_time = 0;
}

void tabucol_show_solution(void) {
  return;
}

/* Number of conflicts for each color and node: (k+1)x(n+1) */
static int **conflicts = NULL;
/* Tabu status for each color and node: (k+1)x(n) */
static int **tabu_status = NULL;
/* Nodes that have conflicts. Position 0 keeps quantity: (k+1)x(n+1) */
static int *nodes_in_conflict = NULL;
/* Index of nodes in array nodes_in_conflicts */
static int *conf_position = NULL;



static void initialize_arrays(gcp_solution_t *solution) {/*{{{*/

  int i, j, n;

  if (nodes_in_conflict == NULL) {
    nodes_in_conflict =(int*) malloc (sizeof(int) * (problem->nof_vertices+1));
    conf_position =(int*) malloc(sizeof(int) * problem->nof_vertices);

    conflicts =(int**) malloc(sizeof(int*) * (problem->max_colors+1));
    tabu_status =(int**) malloc(sizeof(int*) * (problem->max_colors+1));

    for (i = 0; i <= problem->max_colors; i++) {
      conflicts[i] =(int*) malloc(sizeof(int) * (problem->nof_vertices+1));
      tabu_status[i] =(int*) malloc(sizeof(int) * problem->nof_vertices);
    }
  }

  for (i = 0; i <= problem->max_colors; i++) {
    for (j = 0; j < problem->nof_vertices; j++) {
      conflicts[i][j] = 0;
      tabu_status[i][j] = 0;
    }
    conflicts[i][problem->nof_vertices] = 0;
  }

  /* Initializing the conflicts array */
  if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
    for (i = 0; i < problem->nof_vertices; i++) {
      for (j = 1; j <= problem->adj_list[i][0]; j++) {
	n = problem->adj_list[i][j];
	conflicts[solution->color_of[n]][i]++;
      }
    }
  }
  else if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
    for (i = 0; i < problem->nof_vertices; i++) {
      for (j = 0; j < problem->nof_vertices; j++) {
	if (problem->adj_matrix[i * problem->nof_vertices + j])
	  conflicts[solution->color_of[j]][i]++;
      }
    }
  }

}/*}}}*/

static void neighbor_solution(int best_node, int best_color, /*{{{*/
			      gcp_solution_t *solution, int total_it, int t_tenure) {

  int j, last, n;
  int old_color;

  /* Change color of best_node */
  old_color = solution->color_of[best_node];
  solution->color_of[best_node] = best_color;
  if (get_flag(problem->flags, FLAG_S_PARTITION)) {
    for (j = 1; j <= solution->class_color[old_color][0]; j++) {
      if (solution->class_color[old_color][j] == best_node) {
	solution->class_color[old_color][j] =
	  solution->class_color[old_color][solution->class_color[old_color][0]];
	solution->class_color[old_color][0]--;
	break;
      }
    }
    solution->class_color[best_color][0]++;
    solution->class_color[best_color][solution->class_color[best_color][0]] = best_node;
  }

  /* If <old_color> was a conflict and <best_color> is not, remove <best_node>
   * from the list of conflicting nodes */
  if (conflicts[old_color][best_node] && !(conflicts[best_color][best_node])){
    last = nodes_in_conflict[nodes_in_conflict[0]];
    conf_position[last] = conf_position[best_node];
    nodes_in_conflict[conf_position[best_node]] =
      nodes_in_conflict[nodes_in_conflict[0]--];
  }
  else {
    /* If <old_color> was not a conflict and <best_color> is, put
     * <best_node> in the list */
    if (!(conflicts[old_color][best_node]) &&
	conflicts[best_color][best_node]) {
      nodes_in_conflict[0]++;
      conf_position[best_node] = nodes_in_conflict[0];
      nodes_in_conflict[conf_position[best_node]] = best_node;
    }
  }

  /* Update the conflicts of the neighbors */
  if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
    for (j = 1; j <= problem->adj_list[best_node][0]; j++) {

      n = problem->adj_list[best_node][j];

      /* Decrease the number of conflicts in the old color */
      conflicts[old_color][n]--;

      if (conflicts[old_color][n]==0 && solution->color_of[n]==old_color) {
	/* Remove <n> from the list of conflicting nodes if there are 0
	 * conflicts in its own color */
	last = nodes_in_conflict[nodes_in_conflict[0]];
	conf_position[last] = conf_position[n];
	nodes_in_conflict[conf_position[n]] =
	  nodes_in_conflict[nodes_in_conflict[0]--];
      }

      /* Increase the number of conflicts in the new color */
      conflicts[best_color][n]++;

      if (conflicts[best_color][n]==1 && solution->color_of[n]==best_color) {
	/* Add <n> in the list conflicting nodes if there is a new
	 * conflict in its own color */
	nodes_in_conflict[0]++;
	conf_position[n] = nodes_in_conflict[0];
	nodes_in_conflict[conf_position[n]] = n;
      }
    }
  }
  else if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
    for (n = 0; n < problem->nof_vertices; n++) {
      if (problem->adj_matrix[best_node * problem->nof_vertices + n]) {

	/* Decrease the number of conflicts in the old color */
	conflicts[old_color][n]--;

	if (conflicts[old_color][n]==0 && solution->color_of[n]==old_color) {
	  /* Remove <n> from the list of conflicting nodes if there are 0
	   * conflicts in its own color */
	  last = nodes_in_conflict[nodes_in_conflict[0]];
	  conf_position[last] = conf_position[n];
	  nodes_in_conflict[conf_position[n]] =
	    nodes_in_conflict[nodes_in_conflict[0]--];
	}

	/* Increase the number of conflicts in the new color */
	conflicts[best_color][n]++;

	if (conflicts[best_color][n]==1 && solution->color_of[n]==best_color) {
	  /* Add <n> in the list conflicting nodes if there is a new
	   * conflict in its own color */
	  nodes_in_conflict[0]++;
	  conf_position[n] = nodes_in_conflict[0];
	  nodes_in_conflict[conf_position[n]] = n;
	}
      }
    }
  }

  /* Set the tabu status */
  tabu_status[old_color][best_node] = total_it + t_tenure;

}/*}}}*/

static void free_() {/*{{{*/
  int i;
  for (i = 0; i <= problem->max_colors; i++) {
    free(conflicts[i]);
    free(tabu_status[i]);
  }

  free(nodes_in_conflict);
  free(conf_position);
  free(conflicts);
  free(tabu_status);

  conflicts = NULL;
  tabu_status = NULL;
  nodes_in_conflict = NULL;
  conf_position = NULL;

}/*}}}*/

void tabucol(gcp_solution_t *solution, int max_cycles, int type_of_tl) {/*{{{*/

  /* declaring variables *//*{{{*/
  int i, c;
  int pairs[][3] = {
    {10000,10,5}, {10000,15,3}, {10000,5,10}, {5000,15,10},
    {5000,10,15}, {5000,5,20}, {1000,15,30}, {1000,10,50},
    {1000,5,100}, {500,5,100}, {500,10,150}, {500,15,200}
  };

  int num_pairs = sizeof(pairs)/sizeof(int)/3;

  int pair_cycles = 0;
  int frequency = pairs[0][0];
  int increment = pairs[0][1];
  int next_pair = pairs[0][2];

  int total_it = 0;
  int iteration = 0;
  int total_conflicts = 0;
  int best_solution_value;

  int tabu_tenure = problem->nof_vertices/10;

  int min_solution_value = problem->nof_vertices;
  int max_solution_value = 0;

  int nc, best_node, best_color, best_value, node;
  int max_min, p, new_value;

  //printf("TABUCOL\n");

  /*}}}*/

  initialize_arrays(solution);

  /* Count the number of conflicts and set up the list nodes_in_conflict
   * with the associated list conf_position */
  nodes_in_conflict[0] = 0;
  for (i = 0; i < problem->nof_vertices; i++) {
    if (conflicts[solution->color_of[i]][i] > 0) {
      total_conflicts += conflicts[solution->color_of[i]][i];
      nodes_in_conflict[0]++;
      conf_position[i] = nodes_in_conflict[0];
      nodes_in_conflict[conf_position[i]] = i;
    }
  }
  total_conflicts /= 2;

  best_solution_value = total_conflicts;
  if (best_solution_value == 0) {
#if defined TABUCOL
    solution->stop_criterion = STOP_BEST;
#endif
    return;
  }

  if (get_flag(problem->flags, FLAG_VERBOSE) && max_cycles < 0) {
    fprintf(problem->fileout, "Tabucol: arrays initialized; edges in conflict = %d\n, vertices in conflict = %d\n",
	    total_conflicts, nodes_in_conflict[0]);
  }

  while (TRUE) {

    nc = nodes_in_conflict[0];
    total_it++;
    iteration++;

    best_node = -1;
    best_color = -1;
    best_value = problem->nof_vertices * problem->nof_vertices;

    for (i = 1; i <= nodes_in_conflict[0]; i++) {
      node = nodes_in_conflict[i];

      /* Try for every node in conflict to move it to every color */
      for (c = 0; c < problem->max_colors; c++) {
	if (c != solution->color_of[node]) {
	  new_value = total_conflicts + conflicts[c][node]
	    - conflicts[solution->color_of[node]][node];
	  /* Take the new move with minimum value of new conflicts */
	  if (new_value <= best_value) {
	    if ((tabu_status[c][node] < total_it) ||
		(new_value < best_solution_value)) {
	      best_node = node;
	      best_color = c;
	      best_value = new_value;
	    }
	  }
	}
      }
    }

    /* If no non-tabu moves have been found, take any random move */
    if (best_node == -1) {
#if defined LRAND
      //best_node = RANDOM(problem->nof_vertices);
      RANDOM(problem->buffer, best_node, int, problem->nof_vertices);
#elif defined NRAND
      //best_node = RANDOM(problem->seed, problem->nof_vertices);
      RANDOM(problem->seed, problem->buffer, best_node, int, problem->nof_vertices);
#endif
      while (conf_position[best_node] == 0) {
#if defined LRAND
	//best_node = RANDOM(problem->nof_vertices);
	RANDOM(problem->buffer, best_node, int, problem->nof_vertices);
#elif defined NRAND
        //best_node = RANDOM(problem->seed, problem->nof_vertices);
        RANDOM(problem->seed, problem->buffer, best_node, int, problem->nof_vertices);
#endif
      }

#if defined LRAND
      //best_color = RANDOM(problem->max_colors);
      RANDOM(problem->buffer, best_color, int, problem->nof_vertices);
#elif defined NRAND
      //best_color = RANDOM(problem->seed, problem->max_colors);
      RANDOM(problem->seed, problem->buffer, best_color, int, problem->nof_vertices);
#endif

      /* Choose a color that is not the current color of <best_node> but
       * try this for just <|V|> number of tries */
      while (best_color == solution->color_of[best_node]) {
#if defined LRAND
	//best_color = RANDOM(problem->max_colors);
	RANDOM(problem->buffer, best_color, int, problem->nof_vertices);
#elif defined NRAND
        //best_color = RANDOM(problem->seed, problem->max_colors);
	RANDOM(problem->seed, problem->buffer, best_color, int, problem->nof_vertices);
#endif
      }
      best_value = total_conflicts + conflicts[best_color][best_node]
	- conflicts[solution->color_of[best_node]][best_node];
    }

    /* Execute the move */
    neighbor_solution(best_node, best_color, solution, total_it, tabu_tenure); //problem

    total_conflicts = best_value;

    if (get_flag(problem->flags, FLAG_TABUCOL_VERBOSE)) {
      fprintf(problem->fileout, "Tabucol: move executed = (%d,%d); edges in conflict = %d; vertices in conflict = %d\n",
	      best_node+1, best_color, total_conflicts, nodes_in_conflict[0]);
      fprintf(problem->fileout, "\tCycle = %d; best solution so far = %d\n", total_it, best_solution_value);

    }

    /* Calculating tabu_tenure: *//*{{{*/
    if (type_of_tl == TABUCOL_REACTIVE) {
      max_min = 0;
      /* Update the min and max objective function value */
      if (total_conflicts > max_solution_value)
	max_solution_value = total_conflicts;
      if (total_conflicts < min_solution_value)
	min_solution_value = total_conflicts;

      max_min = max_solution_value - min_solution_value;

      if (iteration % frequency == 0) {
	/* Adjust tabu_tenure every <frequency> iterations */
	if ((max_min < (4+total_conflicts/80)) || tabu_tenure == 0) {
	  tabu_tenure += increment;
	  if (pair_cycles == next_pair) {
#if defined LRAND
	    //p = (int) RANDOM(num_pairs);
            RANDOM(problem->buffer, p, int, num_pairs);
#elif defined NRAND
            //p = (int) RANDOM(problem->seed, num_pairs);
            RANDOM(problem->seed, problem->buffer, p, int, num_pairs);
#endif
	    frequency = pairs[p][0];
	    increment = pairs[p][1];
	    pair_cycles = 0;
	    next_pair = pairs[p][2];
	  }
	}
	else if (tabu_tenure) {
	  tabu_tenure--;
	}

	min_solution_value = problem->nof_vertices * problem->nof_vertices;
	max_solution_value = 0;

	if (pair_cycles == next_pair) {
#if defined LRAND
	  //p = RANDOM(num_pairs);
	  RANDOM(problem->buffer, p, int, num_pairs);
#elif defined NRAND
          //p = RANDOM(problem->seed, num_pairs);
          RANDOM(problem->seed, problem->buffer, p, int, num_pairs);
#endif
	  frequency = pairs[p][0];
	  increment = pairs[p][1];
	  pair_cycles = 0;
	  next_pair = pairs[p][2];
	}
	else {
	  pair_cycles++;
	}
      }
    }
    else {
      /* Using a dynamic tabu tenure */
#if defined LRAND
      //tabu_tenure = (int) (0.6 * nc) + RANDOM(10);
      RANDOM(problem->buffer, tabu_tenure, int, 10);
      tabu_tenure = (int) (0.6 * nc) + tabu_tenure;
#elif defined NRAND
      //tabu_tenure = (int) (0.6 * nc) + RANDOM(problem->seed, 10);
      RANDOM(problem->seed, problem->buffer, tabu_tenure, int, 10);
      tabu_tenure = (int) (0.6 * nc) + tabu_tenure;
#endif
    }/*}}}*/

    /* Test if there is a new globally best solution */
    if (total_conflicts <= best_solution_value) {
      best_solution_value = total_conflicts;
#if defined TABUCOL
      solution->cycles_to_best = total_it;
#endif
      solution->time_to_best = current_time_secs(TIME_FINAL, time_initial);

      /* If all nodes are colored, stop iterating */
      if (best_solution_value == 0) {
#if defined TABUCOL
	solution->stop_criterion = STOP_BEST;
#endif
	break;
      }

      /* Reinitialize some values */
      min_solution_value = problem->nof_vertices * problem->nof_vertices;
      max_solution_value = 0;
      iteration = 0;
    }

#if defined TABUCOL
    if (terminate_conditions(solution, total_it, iteration)) {
      break;
    }
#else
    if ((get_flag(problem->flags, FLAG_TIME)) && (problem->max_time <= current_time_secs(TIME_FINAL, time_initial))) {
       break;
   }
    else {
       if ((get_flag(problem->flags, FLAG_TABUCOL_CONV)) && (iteration >= tabucol_info->convergence_cycles)) {
          break;
       }
       else {
          if (total_it >= max_cycles) {
             break;
          }
       }
    }
#endif

  }

  solution->nof_confl_edges = total_conflicts;
  solution->nof_confl_vertices = nodes_in_conflict[0];
  solution->spent_time = current_time_secs(TIME_FINAL, time_initial);
  solution->total_cycles = total_it;

  free_();

}/*}}}*/
