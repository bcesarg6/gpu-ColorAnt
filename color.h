/**********************************************************
 * Created: Seg 08 Ago 2011 16:20:54 BRT
 *
 * Author: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 **********************************************************/
#ifndef __COLOR_H
#define __COLOR_H

#include <stdlib.h>

/* Define kind of stop criterion (to be used) */
#define FLAG_TIME		 1
#define FLAG_CYCLE		 2
#define FLAG_CONV		 4
#define FLAG_TABUCOL_CONV        8
#define FLAG_VERBOSE	        16
#define FLAG_TABUCOL_VERBOSE    32
#define FLAG_COLOR		64
#define FLAG_SEED	       128
#define FLAG_ANTS_RATIO             256

/* Define kind of structure of adjacency */
#define FLAG_ADJ_MATRIX		512
#define FLAG_ADJ_LIST	       1024

/* Define kind of solution: an assignment of colors to vertices or a partition
   of vertices into color classes */
#define FLAG_S_ASSIGN		2048
#define FLAG_S_PARTITION	4096

/* Define the use of memory */
#define FLAG_MEMORY             8192
#define FLAG_MEMORY_REMOVE     16384
#define FLAG_MEMORY_RATIO      32768

/* change the tabucol scheme
 * N cycles of tabucol: N/2 reactive, N/2 dynamic
 */
#define FLAG_CHANGE_TABUCOL_SCHEME 65536

/* apply tabucol in all ants */
#define FLAG_TABUCOL_ALL_ANTS 131072

/* antigo colorant5 */
#define FLAG_CHANGE_ALPHA_BETA 262144

/* antigo colorant4 */
#define FLAG_REUSE_COLOR 524288

#define FLAG_CHANGE_PHEROMONE_SCHEME 1048576

/* change the tabucol scheme
 * N cycles of the whole algorithm: N/2 reactive, N/2 dynamic
 */
#define FLAG_DIFF_TABUCOL_SCHEME 2097152

/* Define kind of stop criterion (when the run is finished) */
#define STOP_CYCLES	1
#define STOP_TIME	2
#define STOP_BEST	3
#define STOP_CONV	4
#define STOP_ALG	5 // specific for some algorithms

/* stopping */
#define DEFAULT_CYCLES                1000
#define DEFAULT_TIME               36000.0
#define DEFAULT_CONVERGENCE_CYCLES     500

#define init_flag(flag)     flag = 0
#define set_flag(flag, pos) flag = flag | pos
#define get_flag(flag, pos) flag & pos

struct gcp_t {
  int nof_vertices;		        // number of vertices
  int ceil_vertices;       //ceil number of vertices in base 2
  int nof_edges;			// number of edges
  int max_cycles;			// maximum number of cycles
  int max_cyc_converg;	                // maximum number of cycles without improvements
  double max_time;	        	// maximum time
  int max_colors;			// maximum of colors (value for k)
  int ceil_colors;           //ceil number of colors in base 2
  int flags;				// flags related to the stop conditions
  int *degree;			        // keeps the degree of each vertex
  int *adj_matrix;         //adjacency matrix
  int **adj_list;			// adjacency list
  double real_time;                     // problem time
#if defined LRAND
  unsigned long int seed;               // seed
#endif
#if defined NRAND
  unsigned short int seed[3];
#endif
  struct drand48_data buffer;
  FILE *fileout;
};

struct gcp_solution_t {
  int *color_of;			// (assignment solution) map vertex->color
  int **class_color;			// (partition solution) color classes
  double spent_time;			// total time spent
  double time_to_best;		        // time to the best so far
  int total_cycles;			// total number of cycles
  int cycles_to_best;			// cycles to the best so far
  int nof_colors;			// number of colors used
  int nof_confl_edges;		        // number of conflicting edges
  int nof_confl_vertices;		// number of conflicting vertices
  int nof_uncolored_vertices;    	// number of uncolored vertices
  int stop_criterion;			// type of stop criterion
  /*
   Funcao de avaliacao baseada no grau

   h1(C) = 1 / |V| * (para todo vertice v com conflito: SUM |Conflicts| * 1/degree(v))
   f1(C) = fc(C) - h1(C) // fc(C) eh a quantidade de conflitos

   Informed Reactive Tabu Search for Graph Coloring
   Daniel Cosmin Porumbel, Jin-kao Hao and Pascale Kuntz
  */
  double h1;
  double f1;
};

typedef struct gcp_t gcp_t;
typedef struct gcp_solution_t gcp_solution_t;

extern gcp_t *problem;

gcp_solution_t* find_solution(void);
void show_solution(gcp_solution_t *solution);
gcp_solution_t* init_solution(void);
int terminate_conditions(gcp_solution_t *solution, int cycle, int converg);
void cpy_solution(gcp_solution_t *src, gcp_solution_t *dst);

#endif /* __COLOR_H */
