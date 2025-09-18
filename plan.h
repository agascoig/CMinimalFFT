
#ifndef __PLAN_H__
#define __PLAN_H__

#include "cminimalfft.h"
#include <stdint.h>

#define P_NONE 0
#define P_INVERSE 1
#define P_INPLACE 2
#define P_REAL 4
#define P_ISBFFT 8
#define P_ODD 16
#define P_SCALED 32
#define P_TOO_MANY_FACTORS 64

#define FUNCODE_DIRECT 8
#define FUNCODE_BLUESTEIN 9
#define DIRECT_SZ 15
#define MAX_FACTORS 3

// Prime factorization result
typedef struct {
  int64_t base[MAX_FACTORS];
  int32_t exponent[MAX_FACTORS];
  int64_t n[MAX_FACTORS];
  int32_t count; // MAX_FACTORS+1 on too many factors
} factorization;

factorization *factorize(int64_t n);

// Inner plan structure
typedef struct {
  int64_t base;
  int64_t ns;
  fft_func_t func;
  int32_t exp;
} inner_plan;

// Minimal plan structure
struct MinimalPlan {
  int64_t n[MAX_REGIONS]; // here: input and output size
  int32_t n_dims;         // number of dimensions
  int32_t region_start;
  int32_t region_end;
  int32_t flags;
  inner_plan ip[MAX_REGIONS][MAX_FACTORS]; // inner plan by region
  int32_t num_factors[MAX_REGIONS]; // number of factors per region
};

void execute_plan(MinimalPlan *P, MFFTELEM **Y, MFFTELEM **X, int64_t r,
                  int64_t bp, int64_t stride);

MinimalPlan *create_min_plan(int64_t *n, int32_t n_dims, int32_t region_start,
                             int32_t region_end, int32_t flags);

void free_min_plan(MinimalPlan *P);

void print_plan(FILE *fp, MinimalPlan *P);

#endif