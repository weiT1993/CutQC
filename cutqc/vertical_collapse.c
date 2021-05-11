#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "mkl.h"

void vertical_collapse(char *subcircuit_kron_terms_file, char *eval_folder, char *eval_mode, char *vertical_collapse_folder, int early_termination, int rank);
void print_float_arr(float *arr, int num_elements);
float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank,int subcircuit_idx);
double get_sec();

int main(int argc, char** argv) {
    int full_circ_size = atoi(argv[1]);
    char *subcircuit_kron_terms_file = argv[2];
    char *eval_folder = argv[3];
    char *vertical_collapse_folder = argv[4];
    int early_termination = atoi(argv[5]);
    int rank = atoi(argv[6]);
    char *eval_mode = argv[7];
    vertical_collapse(subcircuit_kron_terms_file,eval_folder,eval_mode,vertical_collapse_folder,early_termination,rank);
    // printf("%d-q vertical_collapse early_termination %d rank %d DONE\n",full_circ_size,early_termination,rank);
    return 0;
}

void vertical_collapse(char *subcircuit_kron_terms_file, char *eval_folder, char *eval_mode, char *vertical_collapse_folder, int early_termination, int rank) {
    int num_subcircuits, subcircuit_ctr;
    FILE* subcircuit_kron_terms_fptr = fopen(subcircuit_kron_terms_file, "r");
    fscanf(subcircuit_kron_terms_fptr, "%d subcircuits\n", &num_subcircuits);
    double total_collapse_time = 0;
    for (subcircuit_ctr=0;subcircuit_ctr<num_subcircuits;subcircuit_ctr++) {
        int subcircuit_idx, num_kron_terms, num_effective;
        fscanf(subcircuit_kron_terms_fptr, "subcircuit %d kron_terms %d num_effective %d\n", &subcircuit_idx, &num_kron_terms, &num_effective);
        // printf("subcircuit %d kron_terms %d num_effective %d\n", subcircuit_idx, num_kron_terms, num_effective);
        int effective_len;
        if (strcmp(eval_mode,"runtime")==0) {
            effective_len = 1;
        }
        else {
            effective_len = (int) pow(2,num_effective);
        }
        int kron_terms_ctr;
        double subcircuit_collapse_time = 0;
        double log_time = 0;
        for (kron_terms_ctr=0;kron_terms_ctr<num_kron_terms;kron_terms_ctr++) {
            double vertical_collapse_begin = get_sec();
            int subcircuit_kron_index, kron_term_len;
            fscanf(subcircuit_kron_terms_fptr, "subcircuit_kron_index=%d kron_term_len=%d\n", &subcircuit_kron_index, &kron_term_len);
            int subcircuit_inst_ctr;
            float *kron_term = calloc(effective_len,sizeof(float));
            for (subcircuit_inst_ctr=0;subcircuit_inst_ctr<kron_term_len;subcircuit_inst_ctr++) {
                int coefficient, subcircuit_inst_idx;
                fscanf(subcircuit_kron_terms_fptr, "%d,%d ", &coefficient, &subcircuit_inst_idx);
                char *subcircuit_inst_data_file = malloc(256*sizeof(char));
                sprintf(subcircuit_inst_data_file, "%s/measured_%d_%d.txt", eval_folder, subcircuit_idx, subcircuit_inst_idx);
                FILE* subcircuit_inst_data_fptr = fopen(subcircuit_inst_data_file, "r");
                int state_ctr;
                float *subcircuit_inst = calloc(effective_len,sizeof(float));
                for (state_ctr=0;state_ctr<effective_len;state_ctr++) {
                    fscanf(subcircuit_inst_data_fptr, "%f ", &subcircuit_inst[state_ctr]);
                }
                free(subcircuit_inst_data_file);
                fclose(subcircuit_inst_data_fptr);
                // printf("%d ",coefficient);
                // print_float_arr(subcircuit_inst,effective_len);
                // kron_term = coefficient * subcircuit_inst + kron_term
                cblas_saxpy(effective_len,coefficient,subcircuit_inst,1,kron_term,1);
                free(subcircuit_inst);
            }
            bool all_zero = true;
            int state_ctr;
            for (state_ctr=0;state_ctr<effective_len;state_ctr++) {
                if (fabs(kron_term[state_ctr])>1e-16) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero && early_termination==1) {
                free(kron_term);
            }
            else {
                char *subcircuit_kron_term_file = malloc(256*sizeof(char));
                sprintf(subcircuit_kron_term_file, "%s/kron_%d_%d.txt", vertical_collapse_folder, subcircuit_idx, subcircuit_kron_index);
                FILE* subcircuit_kron_term_fptr = fopen(subcircuit_kron_term_file, "w");
                fprintf(subcircuit_kron_term_fptr, "num_effective %d\n", num_effective);
                for (state_ctr=0;state_ctr<effective_len;state_ctr++) {
                    fprintf(subcircuit_kron_term_fptr, "%e ", kron_term[state_ctr]);
                }
                fclose(subcircuit_kron_term_fptr);
                free(subcircuit_kron_term_file);
                free(kron_term);
            }
            log_time += get_sec() - vertical_collapse_begin;
            subcircuit_collapse_time += get_sec() - vertical_collapse_begin;
            log_time = print_log(log_time,subcircuit_collapse_time,kron_terms_ctr+1,num_kron_terms,300,rank,subcircuit_idx);
        }
        total_collapse_time += subcircuit_collapse_time;
    }
    fclose(subcircuit_kron_terms_fptr);
    
    char *summary_file = malloc(256*sizeof(char));
    sprintf(summary_file, "%s/rank_%d_summary.txt", eval_folder, rank);
    FILE *summary_fptr = fopen(summary_file, "a");
    fprintf(summary_fptr,"Total vertical_collapse time = %e\n",total_collapse_time);
    fprintf(summary_fptr,"vertical_collapse DONE\n");
    free(summary_file);
    fclose(summary_fptr);
    return;
}

void print_float_arr(float *arr, int num_elements) {
    int ctr;
    if (num_elements<=10) {
        for (ctr=0;ctr<num_elements;ctr++) {
            printf("%e ",arr[ctr]);
        }
    }
    else {
        for (ctr=0;ctr<5;ctr++) {
            printf("%e ",arr[ctr]);
        }
        printf(" ... ");
        for (ctr=num_elements-5;ctr<num_elements;ctr++) {
            printf("%e ",arr[ctr]);
        }
    }
    printf(" = %d elements\n",num_elements);
}

float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank,int subcircuit_idx) {
    if (log_time>log_frequency) {
        double eta = elapsed_time/num_finished_jobs*num_total_jobs - elapsed_time;
        printf("Rank %d finished subcircuit %d %d/%d, elapsed = %e, ETA = %e\n",rank,subcircuit_idx,num_finished_jobs,num_total_jobs,elapsed_time,eta);
        return 0;
    }
    else {
        return log_time;
    }
}

double get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}