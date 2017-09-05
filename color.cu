/***********************************************************
 * Created: Sex 17 Dez 2015
 *
 * Modificado em 2017
 * Autor: Bruno Cesar Puli Dala Rosa
 *
 * Original version: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 *************************************************************************
 *
 *
 *************************************************************************/
#include <stdio.h>
#include <math.h>
#define _GNU_SOURCE
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include "color.h"
#include "aco.h"
#include "tabucol.h"
#include "util.h"

/* Cuda helper */

static char *namefilein;
gcp_t *problem;
aco_t *aco_info;
tabucol_t *tabucol_info;

void show_help(char *nameprog) {
  printf("Usage: %s [options] <file>\n\n", nameprog);

  printf("ColorAnt options\n");
  printf("  [ -a, --alpha     ] <value>\tDefine alpha parameter as <value>. Default: %d.\n", COLORANT_ALPHA);
  printf("  [ -b, --beta      ] <value>\tDefine beta parameter as <value>. Default: %d.\n", COLORANT_BETA);
  printf("  [ -r, --rho       ] <value>\tDefine rho parameter as <value>. Default: %f.\n", COLORANT_RHO);
  printf("  [ -A, --ants                ] <value>\tDefine number of ants as <value>. In the GPU-VERSIONS they run in parallel. Default: %d.\n", GPU_N_THREADS);
  printf("  [ -R, --use_ants_ratio                   ] \tDefine number of ants as ratio of vertices. Default: FALSE.\n");
  printf("  [ -p, --pheromone-scheme                 ] <value>\tDefine the pheromone scheme. Default: %i.\n\t\t\t\t\t\t\t1: All ants + Best ant + Best colony.\n\t\t\t\t\t\t\t2: Best ant + Best colony\n\t\t\t\t\t\t\t3: Best ant + Best colony (gap).\n", COLORANT_PHEROMONE_SCHEME);
  printf("  [ -n, --change-phero-scheme-iterations   ] <value>\tDefine iterations as <value>. Default: %d without improvement.\n", COLORANT_CHANGE_PHERO_SCHEME_ITERATIONS);

  printf("   Memory Usage:\n\t[ -m, --memory-size                      ] <value>\tDefine memory size as <value>. Default: %d.\n", COLORANT_MEMORY_SIZE);
  printf("\t[ -M, --use-memory-ratio                 ] \tDefine memory size as ratio of ants. Default: FALSE.\n");
  printf("\t[ -d, --delta                            ] <value>\tDefine delta parameter as <value>. Default: %f.\n", COLORANT_DELTA);
  printf("   Pheromone Scheme 3:\n\t[ -g, --gap                              ] <value>\tDefine gap parameter as <value>. Default: %d.\n", COLORANT_GAP);
  printf("   Try to reuse color:\n\t[ -x, --x                                ] <value>\tDefine x parameter as <value>. Default: %f.\n", COLORANT_X);
  printf("\t[ -y, --y                                ] <value>\tDefine y parameter as <value>. Default: %f.\n", COLORANT_Y);

  printf("\nGPU-VERSION options\n");
  printf("  [-z, --gpu_version] Turn on the GPU-COLORANT algorithm. Default: no-gpu-version\n");
  printf("  [-q, --quotient] <value+>\tSet the n_threads/quotient of solutions that is used in the process. Default %d.\n", GPU_APROVEITAMENTO);
  //printf("  [-Z, --nof_threads] <value>  Define number of cuda threads (64,128,256,512,1024,2048,4096,8192). Default: %d.\n", GPU_N_THREADS);

  printf("   Change alpha and beta:\n\t[ -G, --gamma                            ] <value>\tDefine gamma parameter as <value>. Default: %f.\n", COLORANT_GAMMA);
  printf("\t[ -o, --omega                            ] <value>\tDefine omega parameter as <value>. Default: %f.\n", COLORANT_OMEGA);
  printf("\t[ -i, --iterations-alpha-beta            ] <value>\tDefine iterations parameter as <value>. Default: %i.\n", COLORANT_ITERATIONS_ALPHA_BETA);

  printf("\nTabucol options\n");
  printf("\t[ -t, --tabucol-cycles                   ] <value>\tDefine maximum number of cycles in local search as <value>. Default: %d.\n", TABUCOL_CYCLES);
  printf("\t[ -T, --tabucol-convergence-cycles       ] <value>\tDefine maximum number of local search cycles without improvement.Default: %d.\n", TABUCOL_CONVERGENCE_CYCLES);
  printf("\t[ -e, --reactive-scheme                  ] \t\tDefine a reactive scheme for tabu tenure. Default: dynamic scheme.\n");
  printf("\t[ -N, --change-tabucol-scheme-iterations ] <value>\tDefine iterations as <value>. Default: %d.\n", TABUCOL_CHANGE_SCHEME_ITERATIONS);
  printf("\t[ -F, --diff-tabucol-scheme-iterations   ] <value>\tDefine iterations as <value>. Default: %d.\n", TABUCOL_DIFF_SCHEME_ITERATIONS);

  printf("\t[ -u, --apply-tabucol-all-ants                 ] \tApply tabucol on all ants. Default: only on the best ant.\n");

  printf("\nCriterion Stopping\n");
  printf("\t[ -c, --cycles                           ] <value>\tDefine number of cycles.\n");
  printf("\t[ -E, --time                             ] <value> \tDefine time in seconds.\n");
  printf("\t[ -Y, --convergence-cycles               ] <value> \tDefine maximum number of local search cycles without improvement. \n");

  printf("\nGeneral options\n");
  printf("\t[ -k, --colors                           ] <value>\tDefine promote iterations as <value>. Default: vertices.\n");
  printf("\t[ -v, --verbose                          ] \t\tDisplay informations during execution.\n");
  printf("\t[ -V, --tabucol-verbose                  ] \t\tDisplay informations during execution about local search.\n");
  printf("\t[ -s, --seed                             ] <value>\tDefine <value> as the seed of rand function. Default: time\n");
  printf("\t[ -f, --output-filename                  ] <value>\tDefine the output filename. Default: stdout.\n");
  printf("\t[ -h, --help                             ] \t\tDisplay this information.\n\n");
}

void parseargs(int argc, char *argv[]) {

  extern char *optarg;
  char op;

#if defined NRAND
  unsigned long seed;
#endif
  /* Usando getopt para tratamento dos argumentos */
  struct option longopts[] = {
      {"alpha", 1, NULL, 'a'},
      {"beta", 1, NULL, 'b'},
      {"rho", 1, NULL, 'r'},
      {"ants", 1, NULL, 'A'},
      {"use-ants-ratio", 0, NULL, 'R'},
      {"pheromone-scheme", 1, NULL, 'p'},
      {"change-phero-scheme-iterations", 1, NULL, 'n'},
      {"memory-size", 1, NULL, 'm'},
      {"use-memory-mratio", 0, NULL, 'M'},
      {"delta", 1, NULL, 'd'},
      {"gap", 1, NULL, 'g'},
      {"x", 1, NULL, 'x'},
      {"y", 1, NULL, 'y'},
      {"gama", 1, NULL, 'G'},
      {"gpu_version", 2, NULL, 'z'},
      {"quotient_of_use", 2, NULL, 'q'},
      //{"gpu_threads", 1, NULL, 'Z'},
      {"omega", 1, NULL, 'o'},
      {"iterations-alpha-beta", 1, NULL, 'i'},
      {"tabucol-cycles", 1, NULL, 't'},
      {"tabucol-convergence-cycles", 1, NULL, 'T'},
      {"reative-scheme", 0, NULL, 'e'},
      {"change-tabucol-scheme-iterations", 1, NULL, 'N'},
      {"diff-tabucol-scheme-iterations", 1, NULL, 'F'},
      {"apply-tabucol-all-ants", 0, NULL, 'u'},
      {"cycles", 1, NULL, 'c'},
      {"time", 1, NULL, 'E'},
      {"convergence-cycles", 1, NULL, 'Y'},
      {"colors", 1, NULL, 'k'},
      {"verbose", 0, NULL, 'v'},
      {"tabucol-verbose", 0, NULL, 'V'},
      {"seed", 1, NULL, 's'},
      {"output-filename", 1, NULL, 'f'},
      {"help", 0, NULL, 'h'}

  };

  while ((op = getopt_long(argc, argv, "a:b:r:A:Rp:n:m:Md:g:x:y:G:o:i:zqZ:t:T:eN:F:uc:E:Y:k:vVs:f:h", longopts, NULL)) != -1) {

    switch (op) {
      case 'a':
        aco_info->alpha = atof(optarg);
        if (aco_info->alpha <= 0.0)
  	aco_info->alpha = COLORANT_ALPHA;
        break;
      case 'b':
        aco_info->beta = atof(optarg);
        if (aco_info->beta <= 0.0)
  	aco_info->beta = COLORANT_BETA;
        break;
      case 'r':
        aco_info->rho = atof(optarg);
        if (aco_info->rho <= 0.0)
  	aco_info->rho = COLORANT_RHO;
        break;
      case 'A':
        aco_info->nants = atoi(optarg);
        if (aco_info->nants < 1) aco_info->nants = COLORANT_ANTS;
        aco_info->n_threads = aco_info->nants;
        break;
      case 'R':
        set_flag(problem->flags, FLAG_ANTS_RATIO);
        break;
      case 'p':
        aco_info->pheromone_scheme = atoi(optarg);
        if ((aco_info->pheromone_scheme < PHEROMONE_SCHEME_1) || (aco_info->pheromone_scheme > PHEROMONE_SCHEME_3))
  	aco_info->pheromone_scheme = PHEROMONE_SCHEME_1;
        break;
      case 'n':
        aco_info->change_phero_scheme_iterations = atoi(optarg);
        if (aco_info->change_phero_scheme_iterations < 1)
  	aco_info->change_phero_scheme_iterations = COLORANT_CHANGE_PHERO_SCHEME_ITERATIONS;
        set_flag(problem->flags, FLAG_CHANGE_PHEROMONE_SCHEME);
        break;
      case 'm':
        aco_info->memory_size = atoi(optarg);
        if (aco_info->memory_size < 1)
  	aco_info->memory_size = COLORANT_MEMORY_SIZE;
        set_flag(problem->flags, FLAG_MEMORY);
        break;
      case 'M':
        set_flag(problem->flags, FLAG_MEMORY);
        set_flag(problem->flags, FLAG_MEMORY_RATIO);
        break;
      case 'd':
        aco_info->delta = atof(optarg);
        if (aco_info->delta < 0.0)
  	aco_info->delta = COLORANT_DELTA;
        set_flag(problem->flags, FLAG_MEMORY);
        break;
      case 'g':
        aco_info->gap = atoi(optarg);
        if (aco_info->gap < 1)
  	aco_info->gap = COLORANT_GAP;
        break;

      case 'x':
        aco_info->x = atof(optarg);
        if (aco_info->x < 1.0)
  	aco_info->x = COLORANT_X;
        set_flag(problem->flags, FLAG_REUSE_COLOR);
        break;
      case 'y':
        aco_info->y = atof(optarg);
        if (aco_info->y < 1.0)
  	aco_info->y = COLORANT_Y;
        set_flag(problem->flags, FLAG_REUSE_COLOR);
        break;
      case 'G':
        aco_info->gamma = atof(optarg);
        if (aco_info->gamma < 1.0)
  	aco_info->gamma = COLORANT_GAMMA;
        set_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA);
        break;
      case 'o':
        aco_info->omega = atof(optarg);
        if (aco_info->omega < 1.0)
  	aco_info->omega = COLORANT_OMEGA;
        set_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA);
        break;
      case 'i':
        aco_info->iterations_alpha_beta = atoi(optarg);
        if (aco_info->iterations_alpha_beta < 1)
  	aco_info->iterations_alpha_beta = COLORANT_ITERATIONS_ALPHA_BETA;
        set_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA);
        break;

        case 'z':
        aco_info->gpuid = 1;
        break;

        case 'q':
        aco_info->aproveitamento = atoi(optarg);
        break;

        /*case 'Z':
        aco_info->n_threads = atoi(optarg);
        break;*/

      case 't':
        tabucol_info->cycles = atoi(optarg);
        if (tabucol_info->cycles < 1)
  	tabucol_info->cycles = TABUCOL_CYCLES;
        break;
      case 'T':
        tabucol_info->convergence_cycles = atoi(optarg);
        if (tabucol_info->convergence_cycles < 1)
  	tabucol_info->convergence_cycles = TABUCOL_CONVERGENCE_CYCLES;
        set_flag(problem->flags, FLAG_TABUCOL_CONV);
        break;
      case 'e':
        tabucol_info->tl_style = TABUCOL_REACTIVE;
        break;
      case 'N':
        tabucol_info->change_scheme_iterations = atoi(optarg);
        if (tabucol_info->change_scheme_iterations < 1)
  	tabucol_info->change_scheme_iterations = TABUCOL_CHANGE_SCHEME_ITERATIONS;
        set_flag(problem->flags, FLAG_CHANGE_TABUCOL_SCHEME);
        break;
      case 'F':
        tabucol_info->diff_scheme_iterations = atoi(optarg);
        if (tabucol_info->diff_scheme_iterations < 1)
  	tabucol_info->diff_scheme_iterations = TABUCOL_DIFF_SCHEME_ITERATIONS;
        set_flag(problem->flags, FLAG_DIFF_TABUCOL_SCHEME);
        break;
      case 'u':
        set_flag(problem->flags, FLAG_TABUCOL_ALL_ANTS);
        break;
      case 'c':
        problem->max_cycles = atoi(optarg);
        if (problem->max_cycles < 1)
  	problem->max_cycles = DEFAULT_CYCLES;
        set_flag(problem->flags, FLAG_CYCLE);
        break;
      case 'E':
        problem->max_time = atof(optarg);
        if (problem->max_time < 1.0)
  	problem->max_time = DEFAULT_TIME;
        set_flag(problem->flags, FLAG_TIME);
        break;
      case 'Y':
        problem->max_cyc_converg = atoi(optarg);
        if (problem->max_cyc_converg < 1)
  	problem->max_cyc_converg = DEFAULT_CONVERGENCE_CYCLES;
        set_flag(problem->flags, FLAG_CONV);
        break;
      case 'k':
          /*ceil_colors*/
          if (problem->max_colors <= 32){
              problem->ceil_colors = 32;
          } else if (problem->max_colors <= 64){
              problem->ceil_colors = 64;
          } else {
              problem->ceil_colors = 128;
          }

        problem->max_colors = atoi(optarg);
        set_flag(problem->flags, FLAG_COLOR);
        break;

      case 'v':
        set_flag(problem->flags, FLAG_VERBOSE);
        break;
      case 'V':
        set_flag(problem->flags, FLAG_TABUCOL_VERBOSE);
        break;
      case 's':
  #if defined LRAND
        problem->seed = atol(optarg);
  #elif defined NRAND
        seed = atol(optarg);
        memcpy(problem->seed, &seed, sizeof(unsigned short)*3);
  #endif
        set_flag(problem->flags, FLAG_SEED);
        fprintf(stdout, "Sem imprimir flags (%i), está gerando semente ao invés de pegar a passada por parâmetro!!\nSei lá o que está acontecendo!!\nCom este print funciona, então vai assim!!!!\n", problem->flags);
        break;
      case 'f':
        problem->fileout = fopen(optarg, "w");
        break;
      case 'h':
        show_help(argv[0]);
        exit(0);

      }
  }


  /* O único argumento não capturado acima é o nome do arquivo de entrada,
   * se existir */
  if (optind < argc) {
    namefilein =(char*) malloc(sizeof(char) * strlen(argv[optind])+1);
    strcpy(namefilein, argv[optind++]);

    /* verificar se foi passado algum argumento a mais */
    if (optind < argc) {
      printf("error: invalid argument. Use '-h'\n");
      exit(0);
    }
  }
  else {
    printf("error: no input files\n");
    exit(0);
  }

}

void initialization(void){
      FILE *in;

      int i, j, vi, vj;
      char f, t[50];

      in = fopen(namefilein, "r");
      if (!in) {
        printf("error: no input files\n");
        exit(0);
      }

      /* Ignoring initial informations */
      while ((j = fscanf(in, "%c", &f)) && f != 'p') {
        while (f != '\n') {
          j = fscanf(in, "%c", &f);
        }
      }

      j = fscanf(in, "%s %d %d\n", t, &problem->nof_vertices, &problem->nof_edges);
      problem->degree = (int*) malloc(sizeof(int) * problem->nof_vertices);
      if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
        problem->adj_matrix = (int*) malloc(sizeof(int) * problem->nof_vertices * problem->nof_vertices);
      }
      if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
        problem->adj_list = (int**) malloc(sizeof(int*) * problem->nof_vertices);
      }

      for (i = 0; i < problem->nof_vertices; i++) {

        /*if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
          problem->adj_matrix[i] = (char*) malloc(sizeof(char) * problem->nof_vertices);
      }*/
        if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
          problem->adj_list[i] = (int*) malloc(sizeof(int) * (problem->nof_edges+1));
        }

        for (j = 0; j < problem->nof_vertices; j++) {

          if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
    	problem->adj_matrix[i * problem->nof_vertices + j] = 0;
          }
          if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
    	problem->adj_list[i][j] = 0;
          }
        }

        if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
          problem->adj_list[i][problem->nof_vertices] = 0;
        }

        problem->degree[i] = 0;
      }

      for (i = 0; i < problem->nof_edges; i++) {
        j = fscanf(in, "%c %d %d\n", &f, &vi, &vj);

        if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
          problem->adj_matrix[((vi-1) * problem->nof_vertices) + (vj-1)] = 1;    //[vi-1][vj-1]
          problem->adj_matrix[((vj-1) * problem->nof_vertices) + (vi-1)] = 1;    //[vj-1][vi-1]
        }

        if (get_flag(problem->flags, FLAG_ADJ_LIST)) {
          problem->adj_list[vi-1][0]++;
          problem->adj_list[vi-1][problem->adj_list[vi-1][0]] = vj-1;
          problem->adj_list[vj-1][0]++;
          problem->adj_list[vj-1][problem->adj_list[vj-1][0]] = vi-1;
        }

        problem->degree[vi-1]++;
        problem->degree[vj-1]++;
      }

      fclose(in);

      /*ceil_vertices*/
      if(problem->nof_vertices > 1024){

          problem->ceil_vertices = 4096;
      } else{

          problem->ceil_vertices = problem->nof_vertices <= 512 ? 512 : 1024;
      }

}


void printbanner(void) {

#if ! defined TABUCOL
  fprintf(problem->fileout, "-------------------------------------------------\n");
#endif

  if (aco_info->gpuid) gpu_colorant_printbanner();

#if defined COLORANT || defined HCA || defined TABUCOL
  fprintf(problem->fileout, "-------------------------------------------------\n");
  tabucol_printbanner();
#endif

  fprintf(problem->fileout, "-------------------------------------------------\n");
  fprintf(problem->fileout, "GENERAL Options\n");
  fprintf(problem->fileout, "-------------------------------------------------\n");
#if ! defined ANTCOL
  fprintf(problem->fileout, "  Max colors...........: %i\n", problem->max_colors);
  fprintf(problem->fileout, "  Ceil colors (exp 2)..: %i\n", problem->ceil_colors);
#endif
#if defined LRAND
  fprintf(problem->fileout, "  Seed..........................: %lu (lrand)\n", problem->seed);
#elif defined NRAND
  fprintf(problem->fileout, "  Seed..........................: %lu (nrand)\n", print_seed(problem->seed));
#endif
  if (get_flag(problem->flags, FLAG_TIME)) fprintf(problem->fileout, "  Max time......................: %lf\n", problem->max_time);
  if (get_flag(problem->flags, FLAG_CYCLE)) fprintf(problem->fileout, "  Max cycles....................: %d\n", problem->max_cycles);
  if (get_flag(problem->flags, FLAG_CONV)) fprintf(problem->fileout, "  Max cycles without improvement: %d\n", problem->max_cyc_converg);
  if (problem->flags & FLAG_VERBOSE)
    fprintf(problem->fileout, "  Running on Verbose mode.\n");
#if ! defined TABUCOL
  if (problem->flags & FLAG_TABUCOL_VERBOSE )
    fprintf(problem->fileout, "  Running Tabu Search on Verbose mode.\n");
#endif
  fprintf(problem->fileout, "-------------------------------------------------\n");

}

void test_map(gcp_solution_t *solution) {
  int i, j, n;
  int confs = 0;
  for (i = 0; i < problem->nof_vertices; i++) {
    //printf("color of %d: %d\n", i+1, solution->color_of[i]);
    if (get_flag(problem->flags, FLAG_ADJ_MATRIX)) {
      for (j = i; j < problem->nof_vertices; j++) {
	if (problem->adj_matrix[i * problem->nof_vertices + j] &&
	    solution->color_of[i] == solution->color_of[j]) {
	  //printf("ERROR!! Conflicting edge %d--%d \n", i+1, j+1);
	  confs++;
	}
      }
    }
    else {
      for (j = 1; j <= problem->adj_list[i][0]; j++) {
	n = problem->adj_list[i][j];
	if (solution->color_of[i] == solution->color_of[n]) {
	  //printf("ERROR!! Conflicting edge %d--%d \n", i+1, n+1);
	  confs++;
	}
      }
    }
  }
  if (confs != solution->nof_confl_edges) {
    fprintf(problem->fileout, "ERROR!! Confl edges = %d; Calculated = %d\n", confs, solution->nof_confl_edges);
  }
}

void cpy_solution(gcp_solution_t *src, gcp_solution_t *dst) {

  int i, j;
  if (get_flag(problem->flags, FLAG_S_ASSIGN)) {
    for (i = 0; i < problem->nof_vertices; i++) {
      dst->color_of[i] = src->color_of[i];
    }
  }
  if (get_flag(problem->flags, FLAG_S_PARTITION)) {
    for (i = 0; i <= problem->nof_vertices; i++) {
      for (j = 0; j < problem->max_colors; j++)
	dst->class_color[j][i] = src->class_color[j][i];
    }
  }

  dst->spent_time	            = src->spent_time;
  dst->time_to_best	            = src->time_to_best;
  dst->total_cycles	            = src->total_cycles;
  dst->cycles_to_best	      	= src->cycles_to_best;
  dst->nof_colors		        = src->nof_colors;
  dst->nof_confl_edges	     	= src->nof_confl_edges;
  dst->nof_confl_vertices	    = src->nof_confl_vertices;
  dst->nof_uncolored_vertices	= src->nof_uncolored_vertices;
  dst->stop_criterion		    = src->stop_criterion;

}

void show_solution(gcp_solution_t *solution) {
  fprintf(problem->fileout, "\n-------------------------------------------------\n");
  fprintf(problem->fileout, "SOLUTION:\n");
  fprintf(problem->fileout, "-------------------------------------------------\n");
  fprintf(problem->fileout, "No. of colors utilized: %d\n", solution->nof_colors);
  fprintf(problem->fileout, "No. of conflicting edges: %d\n", solution->nof_confl_edges);
  fprintf(problem->fileout, "No. of conflicting vertices: %d\n", solution->nof_confl_vertices);
  fprintf(problem->fileout, "No. of uncolored vertices: %d\n", solution->nof_uncolored_vertices);
  fprintf(problem->fileout, "Real time: %lf\n", problem->real_time);
  fprintf(problem->fileout, "Spent time: %lf\n", solution->spent_time);
#if ! defined CONSTRKGCP
  fprintf(problem->fileout, "Time to the best: %lf\n", solution->time_to_best);
  fprintf(problem->fileout, "Total of cycles: %d\n", solution->total_cycles);
  fprintf(problem->fileout, "Cycles to the best: %d\n", solution->cycles_to_best);
  fprintf(problem->fileout, "Stop criterion: %d\n", solution->stop_criterion);
#endif

#if defined COLORANT
  colorant_show_solution();
#elif defined ANTCOL
  antcol_show_solution();
#elif defined KANTCOL
  kantcol_show_solution();
#elif defined HCA
  hca_show_solution();
#elif defined TABUCOL
  tabucol_show_solution();
#elif defined CONSTRKGCP
  constr_kgcp_show_solution();
#endif

  fprintf(problem->fileout, "-------------------------------------------------\n");
  test_map(solution);
}

/* Essa função nunca é chamada? */
gcp_solution_t* init_solution(void) {
  int i;
  gcp_solution_t *solution;

  solution =(gcp_solution_t*) malloc(sizeof(gcp_solution_t));

  if (get_flag(problem->flags, FLAG_S_ASSIGN)) {
    solution->color_of =(int*) malloc(sizeof(int) * problem->nof_vertices);
  }
  if (get_flag(problem->flags, FLAG_S_PARTITION)) {
    solution->class_color =(int**) malloc(sizeof(int*) * problem->max_colors);
    for (i = 0; i < problem->max_colors; i++) {
      solution->class_color[i] =(int*) malloc(sizeof(int) * (problem->nof_vertices+1));
    }
  }

  solution->nof_colors = 0;
  solution->total_cycles = 0;
  solution->cycles_to_best = 0;
  solution->nof_confl_edges = 0;
  solution->nof_confl_vertices = 0;
  solution->nof_uncolored_vertices = 0;
  solution->stop_criterion = -1;

  return solution;
}

gcp_solution_t* find_solution() {

  gcp_solution_t* sol = NULL;

  if(aco_info->gpuid) sol = gpu_colorant();

  return sol;
}

int terminate_conditions(gcp_solution_t *solution, int cycle, int converg) {

  if (get_flag(problem->flags, FLAG_CONV) &&
      converg >= problem->max_cyc_converg) {
    solution->stop_criterion = STOP_CONV;
    return TRUE;
  }
  else if (get_flag(problem->flags, FLAG_CYCLE) &&
	   cycle >= problem->max_cycles) {
    solution->stop_criterion = STOP_CYCLES;
    return TRUE;
  }
  else if (get_flag(problem->flags, FLAG_TIME) &&
	   current_time_secs(TIME_FINAL, time_initial) >= problem->max_time) {
    solution->stop_criterion = STOP_TIME;
    return TRUE;
  }
  return FALSE;

}

int main(int argc, char *argv[]) {
  gcp_solution_t *results;

  #if defined NRAND
    unsigned long int seed;
    int x;
  #endif

  time_initial = current_time_secs(TIME_INITIAL, 0);

  problem = (gcp_t*) malloc(sizeof(gcp_t));
  init_flag(problem->flags);
  problem->nof_vertices = 0;
  problem->nof_edges = 0;
  problem->max_cycles = 0;
  problem->max_cyc_converg = 0;
  problem->max_time = 0;
  problem->max_colors = 0;
  problem->flags = 0;
  problem->degree = 0;
  problem->adj_matrix = 0;
  problem->adj_list = 0;
  problem->fileout = stdout;
#if defined LRAND
  problem->seed = 0;
#endif
#if defined NRAND
  for (x=0; x<3; x++)
      problem->seed[x] = 0;
#endif

colorant_malloc();
tabucol_malloc();

parseargs(argc, argv);


if (!(get_flag(problem->flags, FLAG_CYCLE)) && !(get_flag(problem->flags, FLAG_TIME))) {
   printf("You need to set the stop criterion.\n");
   exit(0);
}


set_flag(problem->flags, FLAG_S_ASSIGN);
set_flag(problem->flags, FLAG_ADJ_MATRIX);


if(aco_info->gpuid){
    if ((aco_info->aproveitamento == 0 || aco_info->aproveitamento == 1)){
        aco_info->aproveitamento = 1;
    }
    if(!(aco_info->n_threads == 16 || aco_info->n_threads == 64 || aco_info->n_threads == 128 || aco_info->n_threads == 256 || aco_info->n_threads == 512 || aco_info->n_threads == 1024 || aco_info->n_threads == 2048 || aco_info->n_threads ==  4096 || aco_info->n_threads == 8192)){
        printf("Not allowed number of threads, enter 16, 64, 128, 256, 512, 1024, 2048, 4096 or 8192\nAborting.\n");
        exit(0);
    }
}

initialization();

colorant_initialization();
tabucol_initialization();


  if (!(get_flag(problem->flags, FLAG_COLOR))) {
    problem->max_colors = problem->nof_vertices;
  }

  if (!(get_flag(problem->flags, FLAG_SEED))) {
#if defined LRAND
    problem->seed = create_seed();
#endif
#if defined NRAND
    seed = create_seed();
    memcpy(problem->seed, &seed, sizeof(unsigned short)*3);
#endif
    set_flag(problem->flags, FLAG_SEED);
  }

#if defined LRAND
  srand48_r(problem->seed, &problem->buffer);
#endif
#if defined NRAND
  seed48_r(problem->seed, &problem->buffer);
#endif

  printbanner();

 /* if(aco_info->gpuid){
      int devID;
      cudaDeviceProp props;

      // This will pick the best possible CUDA capable device
      devID = findCudaDevice(argc, (const char **)argv);

      //Get GPU information
      checkCudaErrors(cudaGetDevice(&devID));
      checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  }*/

  results = find_solution();

  problem->real_time = current_time_secs(TIME_FINAL, time_initial);

  show_solution(results);

  fclose(problem->fileout);

  return 0;

}
