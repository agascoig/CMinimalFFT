#include "plan.h"
#include "cminimalfft.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Bit test function
static inline bool bt_flags(int32_t flags, int32_t flag) {
  return (flags & flag) != 0;
}

static inline bool bt_plan(MinimalPlan *P, int32_t flag) {
  return bt_flags(P->flags, flag);
}

void print_plan(FILE *fp, MinimalPlan *P) {
  fprintf(fp, "Plan:\n");
  fprintf(fp, "  n_dims: %d\n", P->n_dims);
  fprintf(fp, "  region_start: %d\n", P->region_start);
  fprintf(fp, "  region_end: %d\n", P->region_end);
  fprintf(fp, "  flags: %d\n", P->flags);
  for (int32_t r = P->region_start; r <= P->region_end; r++) {
    fprintf(fp, "  Region %d: n=%lld num_factors=%d\n", r, P->n[r],
            P->num_factors[r]);
    for (int32_t f = 0; f < P->num_factors[r]; f++) {
      inner_plan *ip = &P->ip[r][f];
      fprintf(fp, "    Factor %d: base=%lld exp=%d ns=%lld func=%p\n", f,
              ip->base, ip->exp, ip->ns, (void *)ip->func);
    }
  }
}

static void add_plan_factor(MinimalPlan *P, int32_t region, int64_t ns,
                            int64_t base, int32_t exp, fft_func_t func) {
  int32_t factor_idx = P->num_factors[region];
  inner_plan *s = &(P->ip[region][factor_idx]);

  s->ns = ns;
  s->base = base;
  s->exp = exp;
  s->func = func;

  P->num_factors[region]++;
  minassert(P->num_factors[region] <= MAX_FACTORS,
         "Exceeded maximum factors per region");
}

// Plan 1D FFT
static void plan_1d(MinimalPlan *P, int64_t n, int32_t rd) {
  if (P->num_factors[rd] > 0) {
    return; // region already planned
  }
  minassert(rd <= MAX_REGIONS,"Region index out of bounds");

  factorization *p_factors = factorize(n);

  if (n <= DIRECT_SZ) {
    add_plan_factor(P, rd, n, n, 1, &direct_dft);
  } else if ((n & (n - 1)) == 0) {
    // Power of 2
    int32_t exp = 63 - count_leading_zeros(n);
    add_plan_factor(P, rd, n, 2, exp, &fftr2);
  } else if (p_factors->count <= MAX_FACTORS) {

    for (int32_t i = p_factors->count-1; i >= 0 ; i--) {
      int64_t base = p_factors->base[i];
      int32_t exp = p_factors->exponent[i];
      int32_t nf = p_factors->n[i];
      fft_func_t func = NULL;
      if (nf <= DIRECT_SZ) {
        func = &direct_dft;
      } else {
        func = &bluestein;
        if (base < 10) {
          func = dispatch[base];
          minassert(func != NULL,"dispatch failed.");
        }
      }
      add_plan_factor(P, rd, nf, base, exp, func);
    }
  } else {
    add_plan_factor(P, rd, n, n, 1, &bluestein);
    P->flags |= P_TOO_MANY_FACTORS;
  }

  free(p_factors);
}

static void gen_inner_plan(MinimalPlan *P) {
  for (int64_t r = P->region_start; r <= P->region_end; r++) {
    int64_t nt = P->n[r];
    plan_1d(P, nt, r);
    }
  }

// Execute plan function
void execute_plan(MinimalPlan *P, MFFTELEM **y, MFFTELEM **x, int64_t r,
                  int64_t bp, int64_t stride) {
  bool inverse = bt_plan(P, P_INVERSE);
  int64_t lf = P->num_factors[r];

  if (lf == 0)
    return;

  if (lf == 1) {
    inner_plan *ip1 = &P->ip[r][0];
    ip1->func(y, x, ip1->ns, ip1->exp, bp, stride, inverse ? 1 : 0);
  } else if (lf == 2) {
    inner_plan *ip1 = &P->ip[r][0];
    inner_plan *ip2 = &P->ip[r][1];
    prime_factor_2(y, x, ip1->exp, ip2->exp, ip1->ns, ip2->ns, ip1->func,
                   ip2->func, bp, stride, inverse ? 1 : 0);
  } else if (lf == 3) {
    inner_plan *ip1 = &P->ip[r][0];
    inner_plan *ip2 = &P->ip[r][1];
    inner_plan *ip3 = &P->ip[r][2];
    prime_factor_3(y, x, ip1->exp, ip2->exp, ip3->exp, ip1->ns, ip2->ns, ip3->ns,
                   ip1->func, ip2->func, ip3->func, bp, stride,
                   inverse ? 1 : 0);
  } else {
    minassert(0,"Should have called bluestein with one factor.");
  }
}

// Create minimal plan
MinimalPlan *create_min_plan(int64_t *n, int32_t n_dims, int32_t region_start,
                             int32_t region_end, int32_t flags) {
  MinimalPlan *P = malloc(sizeof(MinimalPlan));

  memcpy(P->n, n, n_dims * sizeof(int64_t));
  P->n_dims = n_dims;
  P->region_start = region_start;
  P->region_end = region_end;
  P->flags = flags;

  for (int32_t i = 0; i < MAX_REGIONS; i++) {
    P->num_factors[i] = 0;
  }

  gen_inner_plan(P);

  return P;
}

// Free minimal plan
void free_min_plan(MinimalPlan *P) {
    free(P);
}