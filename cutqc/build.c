#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include "mkl.h"

#include <immintrin.h>
#include "omp.h"

float* build(char* data_folder, int reconstruction_len, int num_subcircuits, int* subcircuit_indices, int* subcircuit_entry_indices, long long int* subcircuit_prob_lengths);
void print_float_arr(float *arr, long long int num_elements);
void print_int_arr(int *arr, int num_elements);
float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank);
double get_sec();

int main(int argc, char** argv) {
    int rank = atoi(argv[1]);
    char *data_folder = argv[2];
    long long int reconstruction_len = atoi(argv[3]);
    int num_cuts = atoi(argv[4]);
    int num_summation_terms = atoi(argv[5]);
    int num_subcircuits = atoi(argv[6]);
    int num_samples = atoi(argv[7]);
    
    char *build_command_file = malloc(256*sizeof(char));
    sprintf(build_command_file, "%s/build_command_%d.txt", data_folder, rank);
    FILE* build_command_fptr = fopen(build_command_file, "r");

    int summation_term_ctr;
    double total_build_time = 0;
    double log_time = 0;
    float *reconstructed_prob = (float*) calloc(reconstruction_len,sizeof(float));
    for (summation_term_ctr=0; summation_term_ctr<num_summation_terms; summation_term_ctr++) {
        double build_begin = get_sec();
        int subcircuit_ctr;
        int *subcircuit_indices = (int *) calloc(num_subcircuits,sizeof(int));
        int *subcircuit_entry_indices = (int *) calloc(num_subcircuits,sizeof(int));
        long long int *subcircuit_prob_lengths = (long long int *) calloc(num_subcircuits,sizeof(long long int));
        float sampling_prob;
        int frequency;
        fscanf(build_command_fptr,"%f ",&sampling_prob);
        fscanf(build_command_fptr,"%d ",&frequency);
        for (subcircuit_ctr=0; subcircuit_ctr<num_subcircuits; subcircuit_ctr++) {
            fscanf(build_command_fptr,"%d ",&subcircuit_indices[subcircuit_ctr]);
            fscanf(build_command_fptr,"%d ",&subcircuit_entry_indices[subcircuit_ctr]);
            fscanf(build_command_fptr,"%lld ",&subcircuit_prob_lengths[subcircuit_ctr]);
        }
        float* summation_term = build(data_folder, reconstruction_len, num_subcircuits, subcircuit_indices, subcircuit_entry_indices, subcircuit_prob_lengths);
        cblas_sscal(reconstruction_len, frequency/sampling_prob/num_samples, summation_term, 1);
        vsAdd(reconstruction_len, reconstructed_prob, summation_term, reconstructed_prob);
        double build_time = get_sec() - build_begin;
        log_time += build_time;
        total_build_time += build_time;
        if (log_time>300.0) {
            double eta = total_build_time/(summation_term_ctr+1)*num_summation_terms-total_build_time;
            printf("Rank %d built %d/%d summation terms, elapsed = %.3f, ETA = %.3f\n",rank,summation_term_ctr+1,num_summation_terms,total_build_time,eta);
            log_time = 0.0;
        }
    }
    cblas_sscal(reconstruction_len, pow(0.5,num_cuts), reconstructed_prob, 1);
    // print_float_arr(reconstructed_prob,reconstruction_len);

    fclose(build_command_fptr);
    free(build_command_file);

    char *build_file = malloc(256*sizeof(char));
    sprintf(build_file, "%s/build_%d.txt", data_folder, rank);
    FILE *build_fptr = fopen(build_file, "w");
    long long int state_ctr;
    for (state_ctr=0;state_ctr<reconstruction_len;state_ctr++) {
        fprintf(build_fptr,"%e ",reconstructed_prob[state_ctr]);
    }
    fprintf(build_fptr,"\n");
    fclose(build_fptr);
    free(build_file);

    free(reconstructed_prob);
    return 0;
}

float* build(char* data_folder, int reconstruction_len, int num_subcircuits, int* subcircuit_indices, int* subcircuit_entry_indices, long long int* subcircuit_prob_lengths) {
    // Calculate Kronecker product for one summation_term
    // cblas_sger parameters:
    MKL_INT incx, incy;
    CBLAS_LAYOUT layout = CblasRowMajor;
    float alpha = 1;
    incx = 1;
    incy = 1;

    int subcircuit_ctr;
    long long int summation_term_accumulated_len = 0;
    float *summation_term = (float*) calloc(reconstruction_len,sizeof(float));
    for (subcircuit_ctr=0;subcircuit_ctr<num_subcircuits;subcircuit_ctr++) {
        int subcircuit_idx = subcircuit_indices[subcircuit_ctr];
        int subcircuit_entry_idx = subcircuit_entry_indices[subcircuit_ctr];
        long long int subcircuit_prob_length = subcircuit_prob_lengths[subcircuit_ctr];

        char *subcircuit_entry_file = malloc(256*sizeof(char));
        sprintf(subcircuit_entry_file, "%s/%d_%d.txt", data_folder, subcircuit_idx, subcircuit_entry_idx);
        FILE* subcircuit_entry_fptr = fopen(subcircuit_entry_file, "r");

        if (summation_term_accumulated_len==0) {
            long long int state_ctr;
            for (state_ctr=0;state_ctr<subcircuit_prob_length;state_ctr++) {
                fscanf(subcircuit_entry_fptr,"%f ",&summation_term[state_ctr]);
            }
            summation_term_accumulated_len = subcircuit_prob_length;
        }
        else {
            long long int state_ctr;
            float *subcircuit_kron_term = (float*) calloc(subcircuit_prob_length,sizeof(float));
            for (state_ctr=0;state_ctr<subcircuit_prob_length;state_ctr++) {
                fscanf(subcircuit_entry_fptr,"%f ",&subcircuit_kron_term[state_ctr]);
            }
            float *dummy_summation_term = (float*) calloc(summation_term_accumulated_len*subcircuit_prob_length,sizeof(float));
            cblas_sger(layout, summation_term_accumulated_len, subcircuit_prob_length, alpha, summation_term, incx, subcircuit_kron_term, incy, dummy_summation_term, subcircuit_prob_length);
            summation_term_accumulated_len *= subcircuit_prob_length;
            cblas_scopy(summation_term_accumulated_len, dummy_summation_term, 1, summation_term, 1);
            free(dummy_summation_term);
            free(subcircuit_kron_term);
        }
        fclose(subcircuit_entry_fptr);
        free(subcircuit_entry_file);
    }
    return summation_term;
}

void print_int_arr(int *arr, int num_elements) {
    int ctr;
    if (num_elements<=10) {
        for (ctr=0;ctr<num_elements;ctr++) {
            printf("%d ",arr[ctr]);
        }
    }
    else {
        for (ctr=0;ctr<5;ctr++) {
            printf("%d ",arr[ctr]);
        }
        printf(" ... ");
        for (ctr=num_elements-5;ctr<num_elements;ctr++) {
            printf("%d ",arr[ctr]);
        }
    }
    printf(" = %d elements\n",num_elements);
}

void print_float_arr(float *arr, long long int num_elements) {
    long long int ctr;
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
    printf(" = %lld elements\n",num_elements);
}

float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank) {
    if (log_time>log_frequency) {
        double eta = elapsed_time/num_finished_jobs*num_total_jobs - elapsed_time;
        printf("Rank %d finished building %d/%d, elapsed = %e, ETA = %e\n",rank,num_finished_jobs,num_total_jobs,elapsed_time,eta);
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
