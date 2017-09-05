/***********************************************************
 * Created: 19 jun 2017
 *
 * Author: Bruno C.P. Dala Rosa, bcesar.g6@gmail.com
 *
 * GPU_ANT_FIXED_K_B
 * * Parallel Constructive method for n-ants in an ACO algorithm for k-GCP
 *
 ***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

/* CUDA runtime */
#include <cuda_runtime.h>
#include <cuda.h>
/* CUDA cuRand */
#include <curand_kernel.h>
#include <curand.h>

#include "color.h"
#include "aco.h"
#include "util.h"
#include "gpu_ant_fixed_k.cuh"

double *probb;
double *probb_totalsum;
int *vertices;				  /* vertices a serem coloridos*/
int *colors;                  /* cores a serem atribuidas aos respectivo vertices */
int *solutions_color_of;
int *nof_confl_edges;
int *nof_confl_vertices;

/* Device pointers */
//int* d_nof_confl_vertices;
//int* d_solutions_color_of;
int* d_vertices;
int* d_colors;
int* d_vertices_sat;
int* d_neighbors_by_color;
int* d_size_color;
double* d_trail;
double* d_probb;
double* d_probb_totalsum;
int* d_nof_confl_edges;
int* d_confl_vertices;
int* d_conf;

/* Constant device memory */
__constant__ aco_t d_aco_info;

int ants;

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

/* Transfer Constant memory to device */
__host__ void copiaConstant(gcp_t* problem, aco_t* aco_info){
    gpuErrchk(cudaMemcpy(d_problem, problem, sizeof(gcp_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_aco_info, aco_info, sizeof(aco_t)));
}

__global__ void setup_cuRand (curandState* states, unsigned long seed){
  curand_init ( seed, blockIdx.x , 0, &states[blockIdx.x] );
}

/* Choose vertex by Reducing using shared memory */
__global__ void choose_vertex(gcp_t* d_problem, int* d_vertices, int* d_vertices_sat, int* d_solutions_color_of){
	int tid = threadIdx.x;

	/*Vertices ceil values*/
	extern __shared__ short s[];
	short* s_indexes = s;
	short* s_values = &s_indexes[d_problem->ceil_vertices];
    short* s_color_aux = &s_indexes[d_problem->ceil_vertices * 2];

	s_indexes[tid] = tid;
    if(tid < d_problem->nof_vertices){
        s_values[tid] = d_vertices_sat[threadIdx.x + d_problem->nof_vertices * blockIdx.x];
        s_color_aux[tid] = d_solutions_color_of[threadIdx.x + d_problem->nof_vertices * blockIdx.x];
    } else{
        s_color_aux[tid] = 1;
    }

	__syncthreads();

	if (s_color_aux[tid] >= 0 || tid > d_problem->nof_vertices){
        //printf("color_aux = %d\t", d_color_of_aux[globalID]);
		s_values[tid] = -1;
	}
	__syncthreads();

	//do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>=1){
		if (tid < s){
			if ((s_values[tid] < s_values[tid + s]) || (s_values[tid] == s_values[tid + s]) && (s_indexes[tid] > s_indexes[tid + s])){
                //printf("Index %d[%d] = %d[%d]\n",tid, s_values[tid], tid+s, s_values[tid+s]);
				s_values[tid] = s_values[tid + s];
				s_indexes[tid] = s_indexes[tid + s];
			}
		}
		__syncthreads();
	}

	//only thread 0 writes result for this block back to global memory
	if (tid == 0){
		d_vertices[blockIdx.x] = s_indexes[0];
	}
}

__global__ void calculate_probbs(gcp_t* d_problem, double* d_probb, double* d_trail, int* d_size_color, int* d_neighbors_by_color, int* d_vertices, double *d_probb_totalsum){
      int threadID = threadIdx.x;
      int blockID = blockIdx.x;
      double sum, traill, neighbors;
      int size_color, neighbor;

      neighbor   = d_neighbors_by_color[(blockID * d_problem->nof_vertices * d_problem->max_colors) + (d_vertices[blockID] * d_problem->max_colors) + threadID];
      size_color = d_size_color[(blockID *  d_problem->max_colors) + threadID];
      sum        = d_trail[(blockID * d_problem->max_colors * d_problem->nof_vertices) + (threadID * d_problem->nof_vertices) + d_vertices[blockID]]; //Acesso direto a global memory por que só é lido uma vez

      #if defined COLORANT
          if (get_flag(d_problem->flags, FLAG_REUSE_COLOR)) {
              if (size_color == 0) {
                  traill = d_aco_info.y;
              } else {
                  if (neighbor == 0) {
                      traill = d_aco_info.x;
                  } else {
                      traill = sum/size_color;
                  }
              }
          }
      #endif

      traill = (size_color == 0) ? 1 : sum/size_color;
      neighbors = neighbor + 1;
      neighbors = 1.0/neighbors;
      d_probb[(blockID * d_problem->max_colors) + threadID] = pow(traill, (double)d_aco_info.alpha) * pow(neighbors, (double)d_aco_info.beta);
    }

__global__ void probbs_sum(gcp_t* d_problem, double* d_probb, double* d_probb_totalsum){
    int tid = threadIdx.x;
    int max_colors = d_problem->max_colors;

    extern __shared__ double sh[];
    double* s_probb = sh;

    if(tid < max_colors){
        s_probb[tid] = d_probb[tid + max_colors * blockIdx.x];
    } else{
        s_probb[tid] = 0;
    }
    __syncthreads();

    //do reduction in shared memory
    for (unsigned int i = blockDim.x / 2; i > 0; i >>=1){
        if (tid < i && i < max_colors){
            s_probb[tid] += s_probb[tid + i];
        }
        __syncthreads();
    }

    //only thread 0 writes result for this block back to global memory
    if (tid == 0){
        d_probb_totalsum[blockIdx.x] = s_probb[0];
    }
}


/* Usando shared memory e apenas uma thread per block, comparar o desempenho com abordagem global memory e n threads per block */
__global__ void choose_color(gcp_t* d_problem, int *d_colors, curandState_t *states, double *d_probb_totalsum, double *d_probb){
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int i,v;
    double p, last, div;
    last = 0;
    v = 1;

    /* Shared memory setup */
    extern __shared__ double sh[];
    double *s_probb = sh;
    s_probb[threadID] = d_probb[blockID * d_problem->max_colors + threadID];
    __syncthreads();
    /*---------------------*/

    div = d_probb_totalsum[blockID];

    if (threadID == 0){
      p = curand_uniform_double(&states[blockID]);
      //printf("id:%d -> %lf - %lf\n", blockID, div, p); //debugging
      for (i = 0; i < d_problem->max_colors; i++) {
        last += (s_probb[i]/div);
        if (p <= last) {
          v = 0;
          d_colors[blockID] = i;
          i = d_problem->max_colors;
        }
      }
      /* When it reaches here, it means that p == 1 */
      if(v) d_colors[blockID] = d_problem->max_colors -1;
    }
}

__global__ void sol_colors_reset(int* d_solutions_color_of){
    int globalID = threadIdx.x + blockDim.x * blockIdx.x;
    d_solutions_color_of[globalID] = -1;
}

/*Reseta os dados do device que precisam ser resetados */
__host__ void gpu_ant_fixed_k_reset(gcp_t* d_problem){
    sol_colors_reset<<<ants,problem->nof_vertices>>>(d_solutions_color_of);

    gpuErrchk(cudaMemset( d_vertices_sat, 0, sizeof(int) * ants * problem->nof_vertices));
    gpuErrchk(cudaMemset( d_vertices, 0, sizeof(int) * ants));
    gpuErrchk(cudaMemset( d_colors, 0, sizeof(int) * ants));
    gpuErrchk(cudaMemset( d_neighbors_by_color, 0, sizeof(int) * ants * problem->nof_vertices * problem->max_colors));
    gpuErrchk(cudaMemset( d_size_color, 0, sizeof(int) * ants * problem->max_colors));
    gpuErrchk(cudaMemset( d_trail, 0, sizeof(double) * ants * problem->nof_vertices * problem->max_colors));
    gpuErrchk(cudaMemset( d_probb, 0, sizeof(double) * ants * problem->max_colors));
    gpuErrchk(cudaMemset( d_probb_totalsum, 0, sizeof(double) * ants));
    gpuErrchk(cudaMemset( d_nof_confl_edges, 0, sizeof(int) * ants));
    gpuErrchk(cudaMemset( d_nof_confl_vertices, 0, sizeof(int) * ants));
    gpuErrchk(cudaMemset( d_confl_vertices, 0, sizeof(int) * ants * problem->nof_vertices));

    //Host
    memset( vertices, 0, sizeof(int) * ants);
    memset( colors, 0, sizeof(int) * ants);
}

__host__ void ant_fixed_initilization(gcp_t* d_problem){
    ants = aco_info->n_threads;

    probb = (double*) malloc(sizeof(double) * ants * problem->max_colors);                                 //[ants][max_colors]
    probb_totalsum = (double*) malloc(sizeof(double) * ants);                                              //[ants]
    vertices = (int*) malloc(sizeof(int) * ants);                                                          //[ants]
    colors = (int*) malloc(sizeof(int) * ants);                                                            //[ants]
    solutions_color_of = (int*) malloc(sizeof(int) * ants * problem->nof_vertices);                        //[ants][nof_vertices]
    nof_confl_edges = (int*) malloc(sizeof(int) * ants);                                                   //[ants]
    nof_confl_vertices = (int*) malloc(sizeof(int) * ants);                                                //[ants]

    gpuErrchk(cudaMalloc((void **) &d_vertices, sizeof(int) * ants));                                                          //[ants]
    gpuErrchk(cudaMalloc((void **) &d_colors, sizeof(int) * ants));                                                            //[ants]
    gpuErrchk(cudaMalloc((void **) &d_vertices_sat, sizeof(int) * ants * problem->nof_vertices));                              //[ants][nof_vertices] <-> 1024 Testando abordagem de arredondamento!!
    gpuErrchk(cudaMalloc((void **) &d_solutions_color_of, sizeof(int) * ants * problem->nof_vertices));                        //[ants][nof_vertices]; <-> 1024 Testando abordagem de arredondamento!!
    gpuErrchk(cudaMalloc((void **) &d_neighbors_by_color, sizeof(int) * ants * problem->nof_vertices * problem->max_colors));  //[ants][nof_vertices][max_colors]
    gpuErrchk(cudaMalloc((void **) &d_size_color, sizeof(int) * ants * problem->max_colors));                                  //[ants][max_colors]
    gpuErrchk(cudaMalloc((void **) &d_trail, sizeof(double) * ants * problem->max_colors * problem->nof_vertices));            //[ants][max_colors][vertices]
    gpuErrchk(cudaMalloc((void **) &d_probb, sizeof(double) * ants * problem->max_colors));                                    //[ants][max_colors]
    gpuErrchk(cudaMalloc((void **) &d_probb_totalsum, sizeof(double) * ants));                                                 //[ants]
    gpuErrchk(cudaMalloc((void **) &d_nof_confl_edges, sizeof(int) * ants));                                                   //[ants]
    gpuErrchk(cudaMalloc((void **) &d_nof_confl_vertices, sizeof(int) * ants));                                                //[ants]
    gpuErrchk(cudaMalloc((void **) &d_confl_vertices, sizeof(int) * ants * problem->nof_vertices));                            //[ants][nof_vertices]
    gpuErrchk(cudaMalloc((void **) &d_conf, sizeof(int) * ants));                                                              //[ants]
}

/**/
__global__ void ant_fixed_k_update_1(gcp_t* d_problem, int *d_solutions_color_of, int *d_size_color, int *d_vertices, int *d_colors, int *d_nof_confl_edges, int *d_conf){
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;

    d_conf[blockID + threadID] = d_nof_confl_edges[blockID + threadID];
    d_solutions_color_of[((blockID * 8 + threadID) * d_problem->nof_vertices) + d_vertices[blockID * 8 + threadID]] = d_colors[blockID * 8 + threadID];
    d_size_color[((blockID * 8) + threadID) * d_problem->max_colors + d_colors[(blockID * 8) + threadID]]++;
}

__global__ void ant_fixed_k_update_b(gcp_t* d_problem, double *d_trail, int *d_colors, int *d_vertices, int *d_neighbors_by_color,
                                     int *d_solutions_color_of, int *d_nof_confl_edges, int *d_nof_confl_vertices, int *d_confl_vertices,
                                     int *d_size_color, int *d_vertices_sat, int *d_adj_matrix, double *d_pheromone){

    int threadID = threadIdx.x;
    int blockID = blockIdx.x;

    /* trail keeps the pheromone between a vertex and all the vertex already colored with each color */
    d_trail[(blockID * d_problem->max_colors * d_problem->nof_vertices) + (d_colors[blockID] * d_problem->nof_vertices) + threadID] += d_pheromone[d_vertices[blockID] * d_problem->nof_vertices + threadID];
    if (d_adj_matrix[(d_vertices[blockID] * d_problem->nof_vertices) + threadID]){
        /* update degree of saturation: */
        if (d_neighbors_by_color[(blockID * d_problem->nof_vertices * d_problem->max_colors) + (threadID * d_problem->max_colors) + d_colors[blockID]] == 0) {
            d_vertices_sat[(blockID * d_problem->nof_vertices) + threadID]++;
        }
        /* now <i> has a neighbor colored with <color> */
        d_neighbors_by_color[(blockID * d_problem->nof_vertices * d_problem->max_colors) + (threadID * d_problem->max_colors) + d_colors[blockID]]++;

        /* if a neighbor of <v> is colored with <color>, there is a conflicting edge between them */
        if (d_solutions_color_of[(blockID * d_problem->nof_vertices)+ threadID] == d_colors[blockID]){
            d_nof_confl_edges[blockID]++;
            if (d_confl_vertices[(blockID * d_problem->nof_vertices) + threadID] == 0) {
                d_confl_vertices[(blockID * d_problem->nof_vertices) + threadID] = 1;
                d_nof_confl_vertices[blockID]++; // if(threadID == 0)
            }
        }
    }
}

__global__ void ant_fixed_k_update_2(gcp_t* d_problem, int *d_nof_confl_edges, int *d_conf, int *d_confl_vertices, int *d_nof_confl_vertices, int *d_vertices){
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;

    if (d_conf[blockID * 8 + threadID] != d_nof_confl_edges[blockID + threadID]) {
        if (d_confl_vertices[((blockID * 8 + threadID) * d_problem->nof_vertices) +  d_vertices[blockID * 8 + threadID]] == 0) {
            d_confl_vertices[((blockID * 8 + threadID) * d_problem->nof_vertices) +  d_vertices[blockID * 8 + threadID]] = 1;
            d_nof_confl_vertices[blockID * 8 + threadID]++;
        }
    }
}


void print_debug(int *vprint, int tam){
  int i;
  for(i = 0; i < tam; i++){
    printf("%d ", vprint[i]);
  }
  printf("\n");
}

void print_vertices_colors(int *vertices, int *colors, int tam){
    int i;
    for (i = 0; i < tam; i++) {
        printf("vertice %d <- %d color\n", vertices[i], colors[i]);
    }
    printf("\n");
}

/* Processo ant_fixed particionado em vários kernels */
__host__ void gpu_ant_fixed_k (gcp_t* d_problem, gcp_solution_t *solutions, double* d_pheromone, curandState* states, int cycle, int *d_adj_matrix){
    int i;
    int colored = 0;		         /* number of colored vertex */

    /* Reseta a memória do device para começar o ciclo */
    gpu_ant_fixed_k_reset(d_problem);

    /* Times */
    double vertex_time, probbs_time, colors_time, update_time, gpu_ant_fixed_k_time;

    //printf("INFO\nceil_vertices : %d\nmax_colors : %d\n ceil_colors : %d", problem->ceil_vertices, problem->max_colors, problem->ceil_colors);

    while (colored < problem->nof_vertices){
        /* Chose the vertices to be colored */
        vertex_time = current_time_secs(TIME_INITIAL, 0);

        choose_vertex<<<ants, problem->ceil_vertices, sizeof(short) * problem->ceil_vertices * 3>>>(d_problem, d_vertices, d_vertices_sat, d_solutions_color_of);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //debugging vertices
        //gpuErrchk(cudaMemcpy(vertices, d_vertices, sizeof(int) * ants, cudaMemcpyDeviceToHost));
        //printf("Vertices : \n");
        //print_debug(vertices, ants);

        vertex_time = current_time_secs(TIME_FINAL, vertex_time);

        /* Calculate colors probabilities */
        probbs_time = current_time_secs(TIME_INITIAL, 0);

        calculate_probbs<<<ants, problem->max_colors>>>(d_problem, d_probb, d_trail, d_size_color, d_neighbors_by_color, d_vertices, d_probb_totalsum);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        probbs_sum<<<ants, problem->ceil_colors, sizeof(double) * problem->ceil_colors>>>(d_problem, d_probb, d_probb_totalsum);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        probbs_time = current_time_secs(TIME_FINAL, probbs_time);

        /* Choose a color to be assigned to choosen vertices */
        colors_time = current_time_secs(TIME_INITIAL, 0);

        choose_color<<<ants, problem->max_colors, sizeof(double) * problem->max_colors>>>(d_problem, d_colors, states, d_probb_totalsum, d_probb);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        colored++;

        //debugging colors
        //gpuErrchk(cudaMemcpy(colors, d_colors, sizeof(int) * ants, cudaMemcpyDeviceToHost));
        //printf("Colors : \n");
        //print_debug(colors, ants);

        //debugging vertice -> color
        //print_vertices_colors(vertices, colors, ants);

        colors_time = current_time_secs(TIME_FINAL, colors_time);

        /* Update informations about conflicts and saturation degree */
        update_time = current_time_secs(TIME_INITIAL, 0);

        ant_fixed_k_update_1<<<ants/8, 8>>>(d_problem, d_solutions_color_of, d_size_color, d_vertices, d_colors, d_nof_confl_edges, d_conf);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ant_fixed_k_update_b<<<ants, problem->nof_vertices>>>(d_problem, d_trail, d_colors, d_vertices, d_neighbors_by_color, d_solutions_color_of, d_nof_confl_edges,
                                                              d_nof_confl_vertices, d_confl_vertices, d_size_color, d_vertices_sat, d_adj_matrix, d_pheromone);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        ant_fixed_k_update_2<<<ants/8, 8>>>(d_problem, d_nof_confl_edges, d_conf, d_confl_vertices, d_nof_confl_vertices, d_vertices);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        update_time = current_time_secs(TIME_FINAL, update_time);

        /* Soma os times */
        if (problem->flags & FLAG_VERBOSE){
            total_vertex += vertex_time;
            total_probbs += probbs_time;
            total_colors += colors_time;
            total_update += update_time;
        }
    }

    /* Copia as soluções geradas */
    gpuErrchk(cudaMemcpy( nof_confl_edges, d_nof_confl_edges, sizeof(int) * ants, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy( nof_confl_vertices, d_nof_confl_vertices, sizeof(int) * ants, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy( solutions_color_of, d_solutions_color_of, sizeof(int) * ants * problem->nof_vertices, cudaMemcpyDeviceToHost));

    for(i = 0; i < ants; i++){
        solutions[i].nof_colors = problem->max_colors;
        solutions[i].nof_uncolored_vertices = 0;
        solutions[i].total_cycles = cycle;
        solutions[i].nof_confl_edges = nof_confl_edges[i];
        solutions[i].nof_confl_vertices = nof_confl_vertices[i];
        solutions[i].color_of = (int*) malloc(sizeof(int) * problem->nof_vertices);
        memcpy(solutions[i].color_of, &solutions_color_of[i * problem->nof_vertices], sizeof(int) * problem->nof_vertices);
        solutions[i].spent_time = current_time_secs(TIME_FINAL, time_initial);
    }

    /* Reseta a memória do device para o próximo ciclo */
    //gpu_ant_fixed_k_reset();


    /* Implementar SUPER_VERBOSE_FLAG para printar a cada ciclo os tempos
    if (problem->flags & FLAG_S_VERBOSE){
        total_time += total_vertex + total_probbs + total_colors + total_update;
        printf("\n-----------------------------------------------\ngpu_ant_fixed_k total time = %lf\n", total_time);
        printf("     choose_vertex time = %lf\n     calculate_probbs time = %lf\n     choose_color time = %lf\n     update time = %lf\n     reset time  = %lf\n\n", total_vertex, total_probbs, total_colors, total_update, reset_time);
    }*/

    /*FIM*/
}

__host__ void ant_fixed_free(){
    /* Free */
    free(probb);
    free(probb_totalsum);
    free(vertices);
    free(colors);
    free(solutions_color_of);

    /* Cuda Free */
    gpuErrchk(cudaFree(d_vertices));
    gpuErrchk(cudaFree(d_colors));
    gpuErrchk(cudaFree(d_vertices_sat));
    gpuErrchk(cudaFree(d_solutions_color_of));
    gpuErrchk(cudaFree(d_neighbors_by_color));
    gpuErrchk(cudaFree(d_size_color));
    gpuErrchk(cudaFree(d_trail));
    gpuErrchk(cudaFree(d_probb));
    gpuErrchk(cudaFree(d_probb_totalsum));
}
