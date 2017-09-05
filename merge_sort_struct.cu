/********************************************/
/* Autor: Bruno Cesar Puli Dala Rosa        */
/* Ter, 19 de Jan, 2016                     */
/* bcesar.g6@gmail.com                      */
/*                                          */
/* MergeSort para a Struct problem          */
/********************************************/

#include <stdlib.h>
#include <stdio.h>
#include "color.h"

void copy(gcp_solution_t *dst, gcp_solution_t src){
	dst->color_of			= src.color_of;
	dst->spent_time	        = src.spent_time;
    dst->time_to_best	        = src.time_to_best;
    dst->total_cycles	        = src.total_cycles;
    dst->cycles_to_best		= src.cycles_to_best;
    dst->nof_colors		= src.nof_colors;
    dst->nof_confl_edges		= src.nof_confl_edges;
    dst->nof_confl_vertices	= src.nof_confl_vertices;
    dst->nof_uncolored_vertices	= src.nof_uncolored_vertices;
    dst->stop_criterion		= src.stop_criterion;
}

void merge(gcp_solution_t *solutions, int tam){
	if (tam <= 1){
		return;
	}
	gcp_solution_t* vet1,* vet2;
	int i,j,k, tam1, tam2, prox = 0;
	tam1 = tam /2;
	tam2 = (tam % 2) == 0 ? tam/2: (tam/2) + 1;

	gcp_solution_t aux;
	vet1 = (gcp_solution_t*) malloc(tam1 * sizeof(gcp_solution_t));
	vet2 = (gcp_solution_t*) malloc(tam2 * sizeof(gcp_solution_t));

	/* Preenche os dois subvetores */
	j = 0;
	for (i = 0; i < tam; i++) {
		if (i < tam1){
			copy((vet1+i), solutions[i]);
		} else {
		copy((vet2+j), solutions[i]);
		j++;
		}
	}

	merge(vet1, tam1);
	merge(vet2, tam2);

  /* Ordena as metades sucessivamente */
	if(tam < 6){
		for(i = 0; i < (tam1 - 1); i++){
			prox = i+1;
			if(vet1[i].nof_confl_vertices > vet1[prox].nof_confl_vertices){
				aux = vet1[i];
				vet1[i] = vet1[prox];
				vet1[prox] = aux;
			}
		}

		for(i = 0; i < (tam2 - 1); i++){
			prox = i+1;
			if(vet2[i].nof_confl_vertices > vet2[prox].nof_confl_vertices){
				aux = vet2[i];
				vet2[i] = vet2[prox];
				vet2[prox] = aux;
			}
		}
	}

	/* Reconecta os solutions formando o solutions final */
	j = k = i = 0;
	while (i < (tam-1)){
		if (vet1[j].nof_confl_vertices < vet2[k].nof_confl_vertices){
			solutions[i] = vet1[j];
			i++;
			j++;
		} else {
			solutions[i] = vet2[k];
			i++;
			k++;
		}
		if (j == tam1){
			for(k; k < tam2; k++){
				solutions[i] = vet2[k];
				i++;
			}
			break;
		}
		else if (k == tam2){
			for(j; j < tam1; j++){
				solutions[i] = vet1[j];
				i++;
			}
			break;
		}
	}
	free(vet1);
	free(vet2);
	return;
}
