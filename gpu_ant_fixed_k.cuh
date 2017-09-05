/********************************************/
/* Autor: Bruno Cesar Puli Dala Rosa        */
/* 19 jun 2017                              */
/* bcesar.g6@gmail.com                      */
/*                                          */
/*       Header file for gpu_ant_fixed_k_b  */
/********************************************/
#ifndef __GPU_ANT_FIXED_K_B_H
#define __GPU_ANT_FIXED_K__B_H

/* CUDA runtime */
#include <cuda_runtime.h>
#include <cuda.h>
/* CUDA cuRand */
#include <curand_kernel.h>
#include <curand.h>

__global__ void setup_cuRand (curandState_t *states, unsigned long seed);

__host__ void ant_fixed_initilization(gcp_t* d_problem);

__host__ void gpu_ant_fixed_k(gcp_t* d_problem, gcp_solution_t *solutions, double* d_pheromone, curandState_t *states, int cycle, int *d_adj_matrix);

__host__ void ant_fixed_free();

__host__ void copiaConstant (gcp_t* problem, aco_t* aco_info);

__host__ void gpu_ant_fixed_k_reset(gcp_t* d_problem);

__global__ void ant_fixed_k_update_1(gcp_t* d_problem, int *d_solutions_color_of, int *d_size_color, int *d_vertices, int *d_colors, int *d_nof_confl_edges, int *d_conf);

__global__ void ant_fixed_k_update_2(gcp_t* d_problem, int *d_nof_confl_edges, int *d_conf, int *d_confl_vertices, int *d_nof_confl_vertices, int *d_vertices);

__global__ void ant_fixed_k_update_b(gcp_t* d_problem, double *d_trail, int *d_colors, int *d_vertices, int *d_neighbors_by_color,
        int *d_solutions_color_of, int *d_nof_confl_edges, int *d_nof_confl_vertices, int *d_confl_vertices,
        int *d_size_color, int *d_vertices_sat, char *d_adj_matrix, double *d_pheromone);

#endif /* __GPU_ANT_FIXED_K_B_H */
