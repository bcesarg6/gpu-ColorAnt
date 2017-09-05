/***********************************************************
 * Created: Sex 17 Dez 2015
 *
 * Author: Bruno C.P. Dala Rosa, bcesar.g6@gmail.com
 * Original version: Carla N. Lintzmayer, carla0negri@gmail.com
 *
 * GPU VERSAO 1:
 * * Cada thread representa uma formiga e cria uma solução completa.
 * * A quantidade de threads permitida atualmente são apenas valores baixos multiplos de 2
 * * devido ao problema com o estouro de memória.
 * * A quantidade de dados usado no device é grande e por isso grafos grandes
 * * não conseguem ser executados com um numero grande de threads,
 * * a próxima versão possívelmente não terá este problema.
 *
 * VERSAO 1:
 * * Todas as formigas reforçam o feromônio
 * * A melhor formiga da colônia reforça o feromônio
 * * A melhor formiga global reforça o feromônio
 * * A busca local é a React-Tabucol
 * * A busca local é aplicada apenas na melhor formiga da colônia
 *
 * VERSAO 2:
 * * A melhor formiga da colônia reforça o feromônio
 * * A melhor formiga global reforça o feromônio
 * * A busca local é a React-Tabucol
 * * A busca local é aplicada apenas na melhor formiga da colônia
 *
 * VERSAO 3:
 * * A melhor formiga da colônia reforça o feromônio
 * * A melhor formiga global reforça o feromônio
 * * Em cada ciclo, ou a melhor formiga da colônia ou a melhor formiga global
 * * reforçam o feromônio; nunca as duas juntas
 * * A busca local é a React-Tabucol
 * * A busca local é aplicada em todas as formigas da colônia
 *
 * VERSAO 4:
 * * A melhor formiga da colônia reforça o feromônio
 * * A melhor formiga global reforça o feromônio
 * * Em cada ciclo, ou a melhor formiga da colônia ou a melhor formiga global
 * * reforçam o feromônio; nunca as duas juntas
 * * A busca local é a React-Tabucol
 * * A busca local é aplicada em todas as formigas da colônia
 * * O algoritmo tenta reutilizar cores
 *
 * VERSAO 5:
 * * A melhor formiga da colônia reforça o feromônio
 * * A melhor formiga global reforça o feromônio
 * * Em cada ciclo, ou a melhor formiga da colônia ou a melhor formiga global
 * * reforçam o feromônio; nunca as duas juntas
 * * A busca local é a React-Tabucol
 * * A busca local é aplicada em todas as formigas da colônia
 * * O algoritmo ajusta alfa e beta
 *
 *************************************************************************/


 //MUDEI D_PROBLEM DE __CONSTANT__ PARA GLOBAL, VER DESEMPENHO DEPOIS E POSSIVELMENTE MUDAR 19/02/2017
 //MUDAR DE NOVO EXIGE REFATORAÇÃO GRANDE, PESQUISAR SOBRE O DESEMPENHO CONSTANT VS GLOBAL PARA STRUCT PEQUENA

#ifndef __ACO_H
#define __ACO_H

#include "color.h"
#include "tabucol.h"

#define COLORANT_ALPHA            2
#define COLORANT_BETA             8
#define COLORANT_RHO           0.60
#define COLORANT_ANTS          200
#define COLORANT_MEMORY_SIZE           25
#define COLORANT_DELTA          0.5

#define COLORANT_GAP             10
#define COLORANT_X              1.0
#define COLORANT_Y              2.0

#define COLORANT_GAMMA          0.4
#define COLORANT_OMEGA          0.2
#define COLORANT_ITERATIONS_ALPHA_BETA      25
#define COLORANT_PHEROMONE_SCHEME 1
#define COLORANT_CHANGE_PHERO_SCHEME_ITERATIONS 10

/* pheromone scheme 1 */
#define PHEROMONE_SCHEME_1 1

/* pheromone scheme 2 */
#define PHEROMONE_SCHEME_2 2

/* pheromone scheme 3 */
#define PHEROMONE_SCHEME_3 3

/* GPU defines */
#define NOGPU                0
#define GPU1                 1
#define GPU2                 2
#define GPU3                 3
#define GPU_N_THREADS       64
#define GPU_APROVEITAMENTO   1


struct aco_t {
  int gpuid;
  int n_threads;
  int aproveitamento;

  float alpha_base;
  float beta_base;
  float rho_base;
  int iterations_alpha_beta;
  int gap;
  float x;
  float y;
  float gamma;
  float omega;
  int memory_size;
  int memory_ratio;
  float delta;
  int nants;
  int ratio;
  float alpha;
  float beta;
  float rho;

  int pheromone_scheme;
  int change_phero_scheme_iterations;
};

typedef struct aco_memory_t aco_memory_t;

struct aco_memory_t {
  gcp_solution_t *head;
  aco_memory_t *tail;
};

typedef struct aco_t aco_t;

/*Extern declarations */
extern int* d_solutions_color_of;
extern int* d_nof_confl_vertices;
extern int* d_adj_matrix;
extern gcp_t* d_problem;
extern dim3 block,grid;

/*Time gpu_ant_fixed_k */
extern double total_vertex, total_probbs, total_colors, total_update;

extern aco_t *aco_info;

#ifdef COLORANT

void colorant_printbanner(void);
void gpu_colorant_printbanner(void);
void colorant_malloc(void);
void colorant_initialization(void);
void ant_fixed_initilization(void);
void colorant_show_solution(void);
gcp_solution_t* colorant(void);
gcp_solution_t* gpu_colorant(void);


#elif ANTCOL

void antcol_printbanner(void);
void antcol_malloc(void);
void antcol_initialization(void);
void antcol_show_solution(void);
gcp_solution_t* antcol(void);

#elif KANTCOL

void kantcol_printbanner(void);
void kantcol_malloc(void);
void kantcol_initialization(void);
void kantcol_show_solution(void);
gcp_solution_t* kantcol(void);

#endif

#endif /* __COLORANT1_H */
