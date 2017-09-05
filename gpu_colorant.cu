
/***********************************************************
* Created: Sex 17 Dez 2015
*
* Author: Bruno C.P. Dala Rosa, bcesar.g6@gmail.com
* Original version: Carla N. Lintzmayer, carla0negri@gmail.com
*
*************************************************************************
* Versão GPU
* - Esta versão foi implementada a partir da versão base e tem como intuito retirar as alocações dinâmicas dentro do kernel,
*   nela as variáveis locais de cada thread se tornam variáveis globais que englobam o escopo de todas as threads.
*   O acesso a essas variáveis é controlado por uma aritmética de ponteiro que define um intervalo de memória separado em que cada thread irá trabalhar.
*   A alocação de memória foi adaptada.

*************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "aco.h"
#include "tabucol.h"
#include "color.h"
#include "gpu_ant_fixed_k.cuh"

#include "util.h"
#include "merge_sort_struct.h"

/* CUDA runtime */
#include <cuda_runtime.h>
#include <cuda.h>

/* Time vars */
double cycle_time;
double gpu_phero_var_time;

/* Sistema de métrica de tempo de cada função */
double gpu_ant_time, gpu_tabu_time, gpu_pheromone_time;
double media_choose_vertex, media_calculate_probbs, media_choose_colors, media_ant_update;

/* extern definitions <-> aco.h*/
int* d_solutions_color_of;
int* d_nof_confl_vertices;
double total_vertex, total_probbs, total_colors, total_update;
int *d_adj_matrix;
gcp_t* d_problem;

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel launch configuration */
int n_threads;
int coeficiente_aproveitamento;
dim3 block,grid;

/* Global data */
static double *pheromone;
static double *phero_var;
static int gap;
double device_mem_size = 0;

static aco_memory_t *memory = NULL;
static gcp_solution_t *ant_memory_remove;
static gcp_solution_t *ant_memory_insert;

//static gcp_solution_t *ant_k;
static gcp_solution_t *best_colony;
static gcp_solution_t *best_ant;

gcp_solution_t *solutions; //Array de formigas (Soluções)
gcp_solution_t *useful_ants; //Array trimado das melhores soluções. total de formigas / coeficiente_aproveitamento

/* Device pointers */
curandState_t *states;
double *d_pheromone;
int *d_cycle;
double *d_phero_var;
int *d_best_ant_colors;
int *d_best_colony_colors;
int d_best_colony_nof_confl;
int d_best_ant_nof_confl;

/* Device return pointers */
int *solution_color_of;

void gpu_colorant_printbanner() {
    char *schemes[] = {"All ants + Best ant + Best colony", "Best ant + Best colony", "Best ant + Best colony (gap)"};

    fprintf(problem->fileout, "GPU-COLORANT  \n");

    fprintf(problem->fileout, "-------------------------------------------------\n");
    fprintf(problem->fileout, "Graph info:\n");
    fprintf(problem->fileout, "  Number of vertices.....................: %d\n", problem->nof_vertices);
    fprintf(problem->fileout, "  Ceil number of vertices (exp 2)........: %d\n", problem->ceil_vertices);


    fprintf(problem->fileout, "-------------------------------------------------\n");
    fprintf(problem->fileout, "Parameters:\n");

    if(!(aco_info->gpuid)){
        if (!(get_flag(problem->flags, FLAG_ANTS_RATIO))) {
            fprintf(problem->fileout, "  Ants.............................: %i\n", aco_info->nants);
        }
        else {
            fprintf(problem->fileout, "  Ants.............................: %i (%i of %i - vertices)\n", aco_info->nants, aco_info->ratio, problem->nof_vertices);
        }
    } else{
        fprintf(problem->fileout, "  Threads (Ants)...................: %d\n", aco_info->nants);

        if (coeficiente_aproveitamento > 1){
            fprintf(problem->fileout,"  Using only the best %d solutions in the process.\n", n_threads/coeficiente_aproveitamento);
        } else{
            fprintf(problem->fileout,"  Using all the solutions in the process.\n");
        }
    }

    fprintf(problem->fileout, "  Alpha............................: %.2f\n", aco_info->alpha);
    fprintf(problem->fileout, "  Beta.............................: %.2f\n", aco_info->beta);
    fprintf(problem->fileout, "  Rho..............................: %.2f\n", aco_info->rho);

    fprintf(problem->fileout, "  Pheromone scheme.................: %s\n", schemes[aco_info->pheromone_scheme-1]);

    if (get_flag(problem->flags, FLAG_CHANGE_PHEROMONE_SCHEME))
      fprintf(problem->fileout, "  Change pheromone scheme after %i iterations.\n", aco_info->change_phero_scheme_iterations);

      if (get_flag(problem->flags, FLAG_MEMORY)) {
          if (!(get_flag(problem->flags, FLAG_MEMORY_RATIO))) {
             fprintf(problem->fileout, "  Memory Usage:\n\tMemory size......................: %i\n", aco_info->memory_size);
          }
          else {
             fprintf(problem->fileout, "  Memory Usage:\n\tMemory size......................: %i (%i of %i - ants)\n", aco_info->memory_size, aco_info->memory_ratio, aco_info->nants);
          }
        fprintf(problem->fileout, "\tDelta............................: %.2f\n", aco_info->delta);
      }

      if (aco_info->pheromone_scheme == PHEROMONE_SCHEME_3)
      fprintf(problem->fileout, "  Pheromone Scheme 3:\n\tGap..............................: %i\n", aco_info->gap);


      if (get_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA)) {
        fprintf(problem->fileout, "  Change alpha and beta:\n\tGamma............................: %.2f\n", aco_info->gamma);
        fprintf(problem->fileout, "\tOmega............................: %.2f\n", aco_info->omega);
        fprintf(problem->fileout, "\tChange alpha and beta after %i iterations.\n", aco_info->iterations_alpha_beta);
      }
}

void colorant_malloc() {
    aco_info = (aco_t*) malloc(sizeof(aco_t));
    aco_info->pheromone_scheme = PHEROMONE_SCHEME_1;
    aco_info->change_phero_scheme_iterations = COLORANT_CHANGE_PHERO_SCHEME_ITERATIONS;
    aco_info->iterations_alpha_beta = COLORANT_ITERATIONS_ALPHA_BETA;
    aco_info->ratio           = COLORANT_ANTS;
    aco_info->alpha           = COLORANT_ALPHA;
    aco_info->beta            = COLORANT_BETA;
    aco_info->rho             = COLORANT_RHO;
    aco_info->gap             = COLORANT_GAP;
    aco_info->gamma           = COLORANT_GAMMA;
    aco_info->omega           = COLORANT_OMEGA;
    aco_info->x               = COLORANT_X;
    aco_info->y               = COLORANT_Y;
    aco_info->memory_size     = COLORANT_MEMORY_SIZE;
    aco_info->delta           = COLORANT_DELTA;

    aco_info->gpuid           = NOGPU;
    aco_info->nants           = GPU_N_THREADS;
    aco_info->n_threads       = GPU_N_THREADS;
    aco_info->aproveitamento  = GPU_APROVEITAMENTO;

}

void colorant_initialization() {
    aco_info->alpha_base = aco_info->alpha;
    aco_info->beta_base  = aco_info->beta;


    if (get_flag(problem->flags, FLAG_ANTS_RATIO)) {
        aco_info->ratio = aco_info->nants;
        aco_info->nants = (problem->nof_vertices * aco_info->nants) / 100;
    }

    if (get_flag(problem->flags, FLAG_MEMORY_RATIO)) {
          aco_info->memory_ratio = aco_info->memory_size;
          aco_info->memory_size = (aco_info->memory_size * aco_info->nants) / 100;
      aco_info->memory_size = aco_info->memory_size < 1 ? 1 : aco_info->memory_size;
    }
}

void colorant_show_solution() {
     if (get_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA)) {
        fprintf(problem->fileout, "-------------------------------------------------\n");
        fprintf(problem->fileout, "Alpha.: %.2f\n", aco_info->alpha);
        fprintf(problem->fileout, "Beta..: %.2f\n", aco_info->beta);
        fprintf(problem->fileout, "Rho...: %.2f\n", aco_info->rho);
    }
}

static int memory_length() {
  aco_memory_t *item = memory;
  int length = 0;
  for ( ; item != NULL; item = item->tail, length++);
  return length;
}

static void print_memory() {
  aco_memory_t *lmemory = memory;
  int v, count = 1;
  for (; lmemory; lmemory = lmemory->tail) {
    fprintf(problem->fileout, "Item: %i\n", count++);
    fprintf(problem->fileout, "No. of conflicting edges: %d\n", lmemory->head->nof_confl_edges);
    fprintf(problem->fileout, "No. of conflicting vertices: %d\n", lmemory->head->nof_confl_vertices);
    fprintf(problem->fileout, "Color:\n");
    for (v = 0; v < problem->nof_vertices; v++)
      fprintf(problem->fileout, "%i, ", lmemory->head->color_of[v]);
    fprintf(problem->fileout, "\n\n");
  }

  fprintf(problem->fileout, "Removed:\n");
  fprintf(problem->fileout, "No. of conflicting edges: %d\n", ant_memory_remove->nof_confl_edges);
  fprintf(problem->fileout, "No. of conflicting vertices: %d\n", ant_memory_remove->nof_confl_vertices);
  fprintf(problem->fileout, "Color:\n");
  for (v = 0; v < problem->nof_vertices; v++)
    fprintf(problem->fileout, "%i, ", ant_memory_remove->color_of[v]);
  fprintf(problem->fileout, "\n\n");

}

static void insert_into_memory(gcp_solution_t *sol) {
  aco_memory_t *item = (aco_memory_t*) malloc_(sizeof(aco_memory_t*));
  aco_memory_t *last, *previous;

  item->head = sol;
  item->tail = memory;
  memory = item;

  cpy_solution(sol, ant_memory_insert);

  if (aco_info->memory_size < memory_length()) {
    last = memory->tail;
    previous = memory;
    for ( ; last && last->tail != NULL; last = last->tail, previous = previous->tail);
    previous->tail = NULL;
    cpy_solution(last->head, ant_memory_remove);
    set_flag(problem->flags, FLAG_MEMORY_REMOVE);
    free(last);
  }

  //print_memory();

}

/* Inicializa dados locais e do device e copia dados para o device */
__host__ void initialize_data() {
    int i, j;
    n_threads = aco_info->n_threads;
    coeficiente_aproveitamento = aco_info->aproveitamento;

    phero_var = (double*) malloc(problem->nof_vertices * problem->nof_vertices * sizeof(double));

    /* Alocamento de pinned memory para dados que serão transferidos para o device */
    cudaError_t status = cudaMallocHost((void**)&pheromone, problem->nof_vertices * problem->nof_vertices * sizeof(double));
    if (status != cudaSuccess){
        printf("Error allocating pheromone pinned host memory");
        pheromone = (double*) malloc (problem->nof_vertices *problem->nof_vertices* sizeof(double));
    }

    gpuErrchk(cudaMalloc((void **) &d_pheromone, problem->nof_vertices * problem->nof_vertices * sizeof(double)));
    device_mem_size += problem->nof_vertices * problem->nof_vertices * sizeof(double);

    gpuErrchk(cudaMalloc((void **) &d_problem, sizeof(gcp_t)));

    gpuErrchk(cudaMalloc((void **) &d_best_colony_colors, problem->nof_vertices * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &d_best_ant_colors, problem->nof_vertices * sizeof(int)));

    gpuErrchk(cudaMalloc((void**) &d_adj_matrix, problem->nof_vertices * problem->nof_vertices * sizeof(int)));
    device_mem_size += problem->nof_vertices * problem->nof_vertices * sizeof(int);

    gpuErrchk(cudaMemcpy(d_adj_matrix, problem->adj_matrix, problem->nof_vertices * problem->nof_vertices * sizeof(int), cudaMemcpyHostToDevice));
    /*----------------------------------------------------------------------------*/

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = 0; j < problem->nof_vertices; j++) {
            pheromone[i * problem->nof_vertices + j] = 0;
            phero_var[i * problem->nof_vertices + j] = 0;
            if (!problem->adj_matrix[i * problem->nof_vertices + j]) {
                pheromone[i * problem->nof_vertices + j] = 1;
            }
        }
    }

    best_ant = (gcp_solution_t*) malloc (sizeof(gcp_solution_t));
    best_ant->color_of = (int*) malloc (sizeof(int) * problem->nof_vertices);
    best_ant->nof_confl_vertices = INT_MAX;
    best_ant->nof_colors = problem->max_colors;

    best_colony =(gcp_solution_t*) malloc(sizeof(gcp_solution_t));
    best_colony->color_of =(int*) malloc (sizeof(int) * problem->nof_vertices);
    best_colony->nof_confl_vertices = INT_MAX;
    best_colony->nof_colors = problem->max_colors;

    if (get_flag(problem->flags, FLAG_MEMORY)) {
      ant_memory_remove = (gcp_solution_t*)  malloc_(sizeof(gcp_solution_t));
      ant_memory_remove->color_of = (int*) malloc_(sizeof(int) * problem->nof_vertices);
      ant_memory_remove->nof_colors = problem->max_colors;
      ant_memory_remove->spent_time = 0;
      ant_memory_insert = (gcp_solution_t*) malloc_(sizeof(gcp_solution_t));
      ant_memory_insert->color_of = (int*) malloc_(sizeof(int) * problem->nof_vertices);
      ant_memory_insert->nof_colors = problem->max_colors;
      ant_memory_insert->spent_time = 0;

    }

    gpuErrchk(cudaMalloc((void **)&states, aco_info->n_threads * sizeof(curandState_t)));

    gpuErrchk(cudaMalloc((void **)&d_phero_var, problem->nof_vertices * problem->nof_vertices * sizeof(double)));
    gpuErrchk(cudaMemset( d_phero_var, 0, problem->nof_vertices * problem->nof_vertices * sizeof(double)));

    ant_fixed_initilization(d_problem);
    copiaConstant(problem, aco_info);

    setup_cuRand<<<aco_info->n_threads,1>>>(states, problem->seed);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    device_mem_size += n_threads * sizeof(curandState_t);
    device_mem_size += sizeof(gcp_t) + sizeof(aco_t);
}


static void update_pheromone_trails_memory(void) {
 int i, j;

 for (i = 0; i < problem->nof_vertices; i++) {
   for (j = 0; j < problem->nof_vertices; j++) {

     if (!problem->adj_matrix[i * problem->nof_vertices + j]) {
       if (ant_memory_insert->color_of[i] == ant_memory_insert->color_of[j])
         pheromone[i * problem->nof_vertices + j] *=  1 + aco_info->delta;

       if ((get_flag(problem->flags, FLAG_MEMORY_REMOVE)) &&
           (ant_memory_remove->color_of[i] == ant_memory_remove->color_of[j]))
         pheromone[i * problem->nof_vertices + j] *=  1 - aco_info->delta;
     }
   }
 }
}

/* Parallel Functions to help updating pheromone */
__global__ void gpu_update_var_phero(gcp_t* d_problem, double* d_phero_var,
    int* d_solutions_color_of, int *d_nof_confl_vertices, int *d_adj_matrix){

    int i;
    int tid = threadIdx.x;
    int nof_confl_v = d_nof_confl_vertices[0];
    int padding = tid * d_problem->nof_vertices;

    //Use shared memory
    extern __shared__ int s[];
	int* s_solution = s; //nof_vertices

    s_solution[tid] = *(d_solutions_color_of + tid);
    __syncthreads();

    for (i = 0 ; i < d_problem->nof_vertices; i++) {
        if (!d_adj_matrix[padding + i] && (s_solution[i] == s_solution[tid])){
            d_phero_var[padding + i] += nof_confl_v == 0 ? 1 : 1.0/nof_confl_v;
        }
     }
}

/* Scheme 1 */
__global__ void gpu_update_pheromone_trails_colorant1(gcp_t* d_problem,
    double* d_pheromone, double* d_phero_var, int* d_ba_solutions_color_of,
    int* d_bc_solutions_color_of, int ba_nof_confl, int bc_nof_confl,
    int *d_adj_matrix, float rho){

    int i;
    int tid = threadIdx.x;
    int padding = tid * d_problem->nof_vertices;

    //Use shared memory
    extern __shared__ int s[];
	int* ba_solution = s; //nof_vertices
    int* bc_solution = &ba_solution[d_problem->nof_vertices];

    ba_solution[tid] = *(d_ba_solutions_color_of + tid);
    bc_solution[tid] = *(d_bc_solutions_color_of + tid);
    __syncthreads();

    for (i = 0 ; i < d_problem->nof_vertices; i++) {
        d_pheromone[padding + i] += d_phero_var[padding + i];
        d_pheromone[padding + i] *= rho;


        if (!d_adj_matrix[padding + i]){

            if ((ba_solution[i] == ba_solution[tid])){
                d_pheromone[padding + i] += (ba_nof_confl == 0) ? 1 : 1.0/ba_nof_confl;
            }

            if ((bc_solution[i] == bc_solution[tid])){
                d_pheromone[padding + i] += (bc_nof_confl == 0) ? 1 : 1.0/bc_nof_confl;
            }
        }
     }
}

/* Scheme 2 */
__global__ void gpu_update_pheromone_trails_colorant2(gcp_t* d_problem,
    double* d_pheromone, int* d_ba_solutions_color_of,
    int* d_bc_solutions_color_of, int ba_nof_confl, int bc_nof_confl,
    int *d_adj_matrix, float rho){

    int i;
    int tid = threadIdx.x;
    int padding = tid * d_problem->nof_vertices;

    //Use shared memory
    extern __shared__ int s[];
	int* ba_solution = s; //nof_vertices
    int* bc_solution = &ba_solution[d_problem->nof_vertices];

    ba_solution[tid] = *(d_ba_solutions_color_of + tid);
    bc_solution[tid] = *(d_bc_solutions_color_of + tid);
    __syncthreads();

    for (i = 0 ; i < d_problem->nof_vertices; i++) {
        d_pheromone[padding + i] *= rho;

        if (!d_adj_matrix[padding + i]){

            if ((ba_solution[i] == ba_solution[tid])){
                d_pheromone[padding + i] += (ba_nof_confl == 0) ? 1 : 1.0/ba_nof_confl;
            }

            if ((bc_solution[i] == bc_solution[tid])){
                d_pheromone[padding + i] += (bc_nof_confl == 0) ? 1 : 1.0/bc_nof_confl;
            }
        }
     }
}

/* Scheme 3 */
__global__ void gpu_update_pheromone_trails_colorant345(gcp_t* d_problem,
    double* d_pheromone, int* d_ba_solutions_color_of,
    int* d_bc_solutions_color_of, int ba_nof_confl, int bc_nof_confl,
    int *d_adj_matrix, int gap, float rho){

    int i;
    int tid = threadIdx.x;
    int padding = tid * d_problem->nof_vertices;

    //Use shared memory
    extern __shared__ int s[];
	int* ba_solution = s; //nof_vertices
    int* bc_solution = &ba_solution[d_problem->nof_vertices];

    ba_solution[tid] = *(d_ba_solutions_color_of + tid);
    bc_solution[tid] = *(d_bc_solutions_color_of + tid);
    __syncthreads();

    for (i = 0 ; i < d_problem->nof_vertices; i++) {
        d_pheromone[padding + i] *= rho;

        if (!d_adj_matrix[padding + i]){
            if(gap){
                if ((ba_solution[i] == ba_solution[tid])){
                    d_pheromone[padding + i] += (ba_nof_confl == 0) ? 1 : 1.0/ba_nof_confl;
                }

            } else{
                if ((bc_solution[i] == bc_solution[tid])){
                    d_pheromone[padding + i] += (bc_nof_confl == 0) ? 1 : 1.0/bc_nof_confl;
                }
            }
        }
     }
}

/* Functions to help updating pheromone */
__host__ void update_var_phero(gcp_solution_t *solution) {
    int i, j;

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = 0; j < problem->nof_vertices; j++) {
            if (!problem->adj_matrix[i * problem->nof_vertices + j] &&
               (solution->color_of[i] == solution->color_of[j])) {

                phero_var[i * problem->nof_vertices + j] += (solution->nof_confl_vertices == 0) ? 1 : 1.0/solution->nof_confl_vertices;
            }
        }
    }
}

__host__ void update_pheromone_trails_colorant1() {
    int i, j;

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = 0; j < problem->nof_vertices; j++) {
            pheromone[i * problem->nof_vertices + j] += phero_var[i * problem->nof_vertices + j];
            pheromone[i * problem->nof_vertices + j] *= aco_info->rho;

            if (!problem->adj_matrix[i * problem->nof_vertices + j]) {
                if (best_ant->color_of[i] == best_ant->color_of[j]){
                    pheromone[i * problem->nof_vertices + j] += (best_ant->nof_confl_vertices == 0) ? 1 : 1.0/best_ant->nof_confl_vertices;
                }

                if (best_colony->color_of[i] == best_colony->color_of[j]){
                    pheromone[i * problem->nof_vertices + j] += (best_colony->nof_confl_vertices == 0) ? 1 : 1.0/best_colony->nof_confl_vertices;
                }
            }
            phero_var[i * problem->nof_vertices + j] = 0;
        }
    }
}

__host__ void update_pheromone_trails_colorant2() {
    int i, j;

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = 0; j < problem->nof_vertices; j++) {
            pheromone[i * problem->nof_vertices + j] *= aco_info->rho;

            if (!problem->adj_matrix[i * problem->nof_vertices + j]) {
                if (best_ant->color_of[i] == best_ant->color_of[j]) {
                    pheromone[i * problem->nof_vertices + j] += (best_ant->nof_confl_vertices == 0) ? 1 : 1.0/best_ant->nof_confl_vertices;
                }

                if (best_colony->color_of[i] == best_colony->color_of[j]) {
                    pheromone[i * problem->nof_vertices + j] += (best_colony->nof_confl_vertices == 0) ? 1 : 1.0/best_colony->nof_confl_vertices;

                }
            }
        }
    }
}

__host__ void update_pheromone_trails_colorant345(int cycle) {
    int i, j;

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = i; j < problem->nof_vertices; j++) {
            pheromone[i * problem->nof_vertices + j] *= aco_info->rho;

            if (!problem->adj_matrix[i * problem->nof_vertices + j]) {
                if (gap) {
                    if (best_ant->color_of[i] == best_ant->color_of[j])
                    pheromone[i * problem->nof_vertices + j] += (best_ant->nof_confl_vertices == 0) ? 1 : 1.0/best_ant->nof_confl_vertices;
                }
                else {
                    if (best_colony->color_of[i] == best_colony->color_of[j])
                    pheromone[i * problem->nof_vertices + j] += (best_colony->nof_confl_vertices == 0) ? 1 : 1.0/best_colony->nof_confl_vertices;
                }
            }
            pheromone[j * problem->nof_vertices + i] = pheromone[i * problem->nof_vertices + j];
        }
    }
    gap--;
}
/* END Functions to help updating pheromone */

/* Debugging Functions */
void print_solutions(gcp_solution_t* solutions){
    int i;
    for(i = 0; i < n_threads; i++){
        printf("%d - %d\n",i, solutions[i].nof_confl_vertices);
    }
}

void print_phero_var(double* phero_var){
    int i,j;
    int count = 0;

    for (i = 0; i < problem->nof_vertices; i++) {
        for (j = 0; j < problem->nof_vertices; j++) {
            if (phero_var[i * problem->nof_vertices + j] != 0){
                printf("%.3lf  ",phero_var[i * problem->nof_vertices + j] );
                count++;
            }
        }
    }

    printf("PHERO_VAR TOTAL = %d\n", count);
}

/* construct_solutions now will run a kernel launch instead of a for loop! */
static void construct_solutions(int cycle, double *gpu_ant_time, double *gpu_phero_var_time, double *gpu_tabu_time){
    gcp_solution_t *ant_memory;
    int i;
    best_colony->nof_confl_vertices = INT_MAX;

    /* Aloca o useful_ants e o solutions_b */
    useful_ants = (gcp_solution_t*) malloc((n_threads / coeficiente_aproveitamento) * sizeof(gcp_solution_t));
    solutions = (gcp_solution_t*) malloc(sizeof(gcp_solution_t) * n_threads);

    /* Transfere o pheromone para a global memory */
    gpuErrchk(cudaMemcpy(d_pheromone, pheromone, problem->nof_vertices * problem->nof_vertices * sizeof(double), cudaMemcpyHostToDevice));

    *gpu_ant_time = current_time_secs(TIME_INITIAL, 0);

    /* gpu_ant_fixed_k launch */
    gpu_ant_fixed_k(d_problem, solutions, d_pheromone, states, cycle, d_adj_matrix);


    *gpu_ant_time = current_time_secs(TIME_FINAL, *gpu_ant_time); //tempo do ant_fixed_k
    //printf("\nCiclo: %d Pre-ordenação:\n",cycle);
    //print_solutions(solutions);

    /* TABU-SEARCH EM TODAS SOL  Antigos color(3 4 5) */
    if ((get_flag(problem->flags, FLAG_TABUCOL_ALL_ANTS) && (useful_ants[0].nof_confl_vertices != 0) && (tabucol_info->cycles > 0))) {
        *gpu_tabu_time = current_time_secs(TIME_INITIAL, 0);
        for(i = 0; i < n_threads / coeficiente_aproveitamento; i++){
            tabucol(solutions+i, tabucol_info->cycles, tabucol_info->tl_style);
        }

        *gpu_tabu_time = current_time_secs(TIME_FINAL, *gpu_tabu_time);
    }

    /* Ordena o array de soluções em ordem crescente de nof_confl_vertices*/
    merge(solutions,n_threads);

    /* Trima o array de soluções para o tamanho definido pelo coeficiente de aproveitamento das soluções */
    if(coeficiente_aproveitamento > 1){
        memcpy(useful_ants, solutions, (n_threads / coeficiente_aproveitamento) * sizeof(gcp_solution_t));
    } else {
        useful_ants = solutions;
    }

    if (aco_info->pheromone_scheme == PHEROMONE_SCHEME_1){
        *gpu_phero_var_time = current_time_secs(TIME_INITIAL, 0);
        for (i = 0; i < n_threads / coeficiente_aproveitamento; i++){
            //update_var_phero(useful_ants + i); // Método sequencial
            gpu_update_var_phero<<<1,problem->nof_vertices,problem->nof_vertices * sizeof(int)>>>(d_problem, d_phero_var, &d_solutions_color_of[i * problem->nof_vertices], &d_nof_confl_vertices[i], d_adj_matrix);
        }
        *gpu_phero_var_time = current_time_secs(TIME_FINAL, *gpu_phero_var_time);

        //debugging print
        gpuErrchk(cudaMemcpy(phero_var, d_phero_var, problem->nof_vertices * problem->nof_vertices * sizeof(double), cudaMemcpyDeviceToHost));
        //printf("PRINT phero_var\n");
        //print_phero_var(phero_var);
    }

    /* BUSCA-TABU APENAS NA MELHOR SOLUÇÃO Antigo color (1 2)*/
    if ((!(get_flag(problem->flags, FLAG_TABUCOL_ALL_ANTS))) && (useful_ants[0].nof_confl_vertices != 0) && (tabucol_info->cycles > 0)) {
        *gpu_tabu_time = current_time_secs(TIME_INITIAL, 0);

        tabucol(useful_ants, tabucol_info->cycles, tabucol_info->tl_style);
        *gpu_tabu_time = current_time_secs(TIME_FINAL, *gpu_tabu_time);
    }

    //printf("\nCiclo: %d Pos-Tabu:\n",cycle);
    //print_solutions(solutions);

    /* Best colony sempre será a formiga da posição 0 do vetor de soluções pós-sort */
    cpy_solution(useful_ants, best_colony);
    best_colony->cycles_to_best = cycle;
    best_colony->time_to_best = useful_ants[0].spent_time;

    gpuErrchk(cudaMemcpy(d_best_colony_colors, best_colony->color_of, problem->nof_vertices * sizeof(int), cudaMemcpyHostToDevice));
    d_best_colony_nof_confl = best_colony->nof_confl_vertices;

    if(coeficiente_aproveitamento > 1 ) free(solutions);
    free(useful_ants);

    if (get_flag(problem->flags, FLAG_MEMORY)) {
      ant_memory = (gcp_solution_t*) malloc_(sizeof(gcp_solution_t));
      ant_memory->color_of = (int*) malloc_(sizeof(int) * problem->nof_vertices);
      cpy_solution(best_colony, ant_memory);
      insert_into_memory(ant_memory);
    }
}


gcp_solution_t* gpu_colorant() {
    int cycle = 0;
    int converg = 0;
    int change = 0;

    double media_gpu_ant = 0;
    double media_uvar_phero = 0;
    double media_tabu = 0;
    double media_pheromone = 0;
    double media_cycle = 0;

    initialize_data();
    best_ant->stop_criterion = 0;

    //printf("%f\n", aco_info->delta);
    if (problem->flags & FLAG_VERBOSE){
        fprintf(problem->fileout,"Total memory used on Device: %.4lf KBytes\n\n", device_mem_size/1024.);
    }

    while (!terminate_conditions(best_ant, cycle, converg)) {
        cycle_time = current_time_secs(TIME_INITIAL, 0);
        cycle++;
        converg++;

        construct_solutions(cycle, &gpu_ant_time, &gpu_phero_var_time, &gpu_tabu_time);

        gpu_pheromone_time = current_time_secs(TIME_INITIAL, 0);
        if (best_colony->nof_confl_vertices < best_ant->nof_confl_vertices) {
            cpy_solution(best_colony, best_ant);
            best_ant->cycles_to_best = cycle;
            best_ant->time_to_best = best_colony->spent_time;
            converg = 0;
            change = 1;

            if(aco_info->gpuid){
                gpuErrchk(cudaMemcpy(d_best_ant_colors, best_colony->color_of, problem->nof_vertices * sizeof(int), cudaMemcpyHostToDevice));
                d_best_ant_nof_confl = d_best_colony_nof_confl;
            }
        }


        switch (aco_info->pheromone_scheme) {
            case PHEROMONE_SCHEME_1:
            //update_pheromone_trails_colorant1();

            gpu_update_pheromone_trails_colorant1<<<1,problem->nof_vertices,problem->nof_vertices * sizeof(int) * 2>>>(d_problem,
                d_pheromone, d_phero_var, d_best_ant_colors, d_best_colony_colors, best_ant->nof_confl_vertices,
                best_colony->nof_confl_vertices, d_adj_matrix, aco_info->rho);

            gpuErrchk(cudaMemset( d_phero_var, 0, problem->nof_vertices * problem->nof_vertices * sizeof(double)));
            break;

            case PHEROMONE_SCHEME_2:
            //update_pheromone_trails_colorant2();

            gpu_update_pheromone_trails_colorant2<<<1,problem->nof_vertices,problem->nof_vertices * sizeof(int) * 2>>>(d_problem,
                d_pheromone, d_best_ant_colors, d_best_colony_colors, best_ant->nof_confl_vertices,
                best_colony->nof_confl_vertices, d_adj_matrix, aco_info->rho);
            break;

            case PHEROMONE_SCHEME_3:
            if (cycle % aco_info->gap == 0) gap = cycle / aco_info->gap;

            //update_pheromone_trails_colorant345(cycle);

            gpu_update_pheromone_trails_colorant345<<<1,problem->nof_vertices,problem->nof_vertices * sizeof(int) * 2>>>(d_problem,
                d_pheromone, d_best_ant_colors, d_best_colony_colors, best_ant->nof_confl_vertices,
                best_colony->nof_confl_vertices, d_adj_matrix, gap, aco_info->rho);

            gap--;
            break;
        }


        if (get_flag(problem->flags, FLAG_MEMORY)) {
            gpuErrchk(cudaMemcpy(pheromone, d_pheromone, problem->nof_vertices * problem->nof_vertices * sizeof(double), cudaMemcpyDeviceToHost));

            update_pheromone_trails_memory();

            gpuErrchk(cudaMemcpy(d_pheromone, pheromone, problem->nof_vertices * problem->nof_vertices * sizeof(double), cudaMemcpyHostToDevice));

        }

        //debugging print
        //printf("PRINT pheromone\n");
        //print_phero_var(pheromone);
        gpu_pheromone_time = current_time_secs(TIME_FINAL, gpu_pheromone_time); //tempo do update pheromone

        if (best_ant->nof_confl_vertices == 0) {
            best_ant->nof_uncolored_vertices = 0;
            best_ant->stop_criterion = STOP_BEST;
            break;
        }

        if ( get_flag(problem->flags, FLAG_CHANGE_ALPHA_BETA) && ((cycle % aco_info->iterations_alpha_beta)==0) ) {

            aco_info->gamma = change ? (1 - aco_info->omega) * aco_info->gamma:
            (1 + aco_info->omega) * aco_info->gamma;

            aco_info->alpha = aco_info->alpha_base * aco_info->gamma;
            aco_info->beta  = aco_info->beta_base  * (1 - aco_info->gamma);

            if ( (aco_info->alpha < 0) && ( aco_info->beta > 0) )
            aco_info->gamma = (1 + aco_info->omega) * aco_info->gamma;
            else
            if ( (aco_info->alpha > 0) && (aco_info->beta < 0) )
            aco_info->gamma = (1 - aco_info->omega) * aco_info->gamma;

            aco_info->alpha = aco_info->alpha_base * aco_info->gamma;
            aco_info->beta  = aco_info->beta_base  * (1 - aco_info->gamma);

            change = 0;

            //printf("2 alfa: %.2f beta:%.2f gama:%.2f omega:%.2f\n", aco_info->alpha, aco_info->beta, aco_info->gamma, aco_info->omega);
        }

        cycle_time = current_time_secs(TIME_FINAL, cycle_time);//tempo total do ciclo

        if (problem->flags & FLAG_VERBOSE) {
            fprintf(problem->fileout, "\nCycle %d - Conflicts found: %d (edges), %d (vertices)\n", cycle, best_ant->nof_confl_edges, best_ant->nof_confl_vertices);
            fprintf(problem->fileout, "Tempo do método ant_fixed_k: %lf\n", gpu_ant_time);
            fprintf(problem->fileout, "Tempo do ciclo: %lf\n", cycle_time);

        }

        /* Soma medias de tempo */
        media_gpu_ant += gpu_ant_time;
        media_uvar_phero += gpu_phero_var_time;
        media_tabu += gpu_tabu_time;
        media_pheromone += gpu_pheromone_time;
        media_cycle += cycle_time;
    }

    if (problem->flags & FLAG_VERBOSE){
        media_choose_vertex    = total_vertex    / cycle;
        media_calculate_probbs = total_probbs    / cycle;
        media_choose_colors    = total_colors    / cycle;
        media_ant_update       = total_update    / cycle;
        media_cycle            = media_cycle     / cycle;
        media_gpu_ant          = media_gpu_ant   / cycle;
        media_tabu             = media_tabu      / cycle;
        media_pheromone        = media_pheromone / cycle;

        fprintf(problem->fileout, "\n===\nMedias de tempo dos Kernels\n===\n");
        fprintf(problem->fileout, "Choose_vertex    : %lf\nCalculate_probbs : %lf\nChoose_colors    : %lf\nUpdate_info      : %lf\n", media_choose_vertex, media_calculate_probbs, media_choose_colors, media_ant_update);
        fprintf(problem->fileout, "\n===\nMedias de tempo dos métodos\n===\n");
        fprintf(problem->fileout, "Media ciclos     : %lf\nMedia gpu_ant_k  : %lf\nMedia busca tabu : %lf\nMedia feromonio  : %lf\n Media Phero_var  : %lf\n", media_cycle, media_gpu_ant, media_tabu, media_pheromone, media_uvar_phero);
    }

    best_ant->spent_time = current_time_secs(TIME_FINAL, time_initial);
    best_ant->total_cycles = cycle;
    return best_ant;
}
