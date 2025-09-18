
#ifndef CMINIMALFFT_H
#define CMINIMALFFT_H

#include <assert.h>
#include <complex.h>
#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __BSD_VISIBLE 1
#include <openlibm_math.h>

// Define the element type
typedef double complex MFFTELEM;

static inline double *real(complex double *z) { return (double *)z; }

static inline double *imag(complex double *z) { return ((double *)z) + 1; }

// Completely disable any builtin sincos
#ifdef sincos
#undef sincos
#endif

// Declare sincos from OpenLibm
void sincos(double x, double *sin, double *cos);

static inline void minsincos(double angle, double *sinp, double *cosp) {
  sincos(angle, sinp, cosp);
}

static inline complex double times_pmim(complex double z, int inverse) {
  if (inverse) {
    return ((-__imag__(z)) + (__real__(z)) * I);
  } else {
    return (__imag__(z) + (-__real__(z)) * I);
  }
}

static void swap_ptrs(MFFTELEM **a, MFFTELEM **b) {
  MFFTELEM *temp = *a;
  *a = *b;
  *b = temp;
}

static void print_stacktrace(void) {
  void *buffer[100];
  int nptrs = backtrace(buffer, 100);
  char **strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Stacktrace:\n");
  for (int i = 0; i < nptrs; i++) {
    fprintf(stderr, "%s\n", strings[i]);
  }
  free(strings);
}

#define minassert(cond, msg)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "Assertion failed: %s\n", msg);                          \
      print_stacktrace();                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static inline void *minaligned_alloc(size_t alignment, size_t sz,
                                     size_t count) {
  void *p = aligned_alloc(alignment, sz * count);
  minassert(p, "Memory allocation failed.");
  return p;
}

static inline void *minaligned_calloc(size_t alignment, size_t sz,
                                      size_t count) {
  void *p = aligned_alloc(alignment, sz * count);
  if (!p) {
    print_stacktrace();
  }
  minassert(p, "Memory allocation failed.");
  memset(p, 0, sz * count);
  return p;
}

static int approx_cmp(MFFTELEM x, MFFTELEM y) {
  // borrowed from Julia for double
  double atol = 0;
  double rtol = 1.4901161193847656e-8; // sqrt(eps(Float64))

  if (x == y)
    return 0;

  // Check for finite values
  if (isfinite(creal(x)) && isfinite(cimag(x)) && isfinite(creal(y)) &&
      isfinite(cimag(y))) {
    double diff = cabs(x - y);
    double norm_x = cabs(x);
    double norm_y = cabs(y);
    double tol = fmax(atol, rtol * fmax(norm_x, norm_y));
    if (diff <= tol)
      return 0;
  }

  return 1;
}

static double norm_v(MFFTELEM *x, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += creal(x[i] * conj(x[i]));
  }
  return sqrt(sum);
}

static int is_finite(MFFTELEM *x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(creal(x[i])) || !isfinite(cimag(x[i]))) {
      return 0;
    }
  }
  return 1;
}

static int approx_cmp_v(MFFTELEM *x, MFFTELEM *y, size_t n) {
  double atol = 0;
  double rtol = 1.4901161193847656e-8; // sqrt(eps(Float64))

  if (x == y)
    return 0;

  // Check for finite values
  if (is_finite(x, n) && is_finite(y, n)) {
    double diff = 0.0;
    for (int i = 0; i < n; ++i) {
      complex double c = x[i] - y[i];
      diff += creal(c * conj(c));
    }
    diff = sqrt(diff);
    double norm_x = norm_v(x, n);
    double norm_y = norm_v(y, n);
    double tol = fmax(atol, rtol * fmax(norm_x, norm_y));
    if (diff <= tol)
      return 0;
  }

  return 1;
}

#define MAX_DIMS 8
#define MAX_REGIONS MAX_DIMS

// Structure to represent multi-dimensional or decomposed array info
typedef struct {
  MFFTELEM *data;
  int64_t dims[MAX_DIMS];
  int64_t total_size;
  int32_t ndims;
} MDArray;

static MDArray create_mdarray(MFFTELEM *data, int64_t *dims, int32_t ndims) {
  MDArray arr;
  arr.data = data;
  memcpy(arr.dims, dims, ndims * sizeof(int64_t));
  arr.ndims = ndims;

  arr.total_size = 1;
  for (int64_t i = 0; i < ndims; i++) {
    arr.total_size *= dims[i];
  }

  return arr;
}

static void free_mdarray(MDArray *arr) {
  free(arr); // do not free the data itself
}

typedef void (*fft_func_t)(MFFTELEM **Y, MFFTELEM **X, int64_t N, int32_t e1,
                           int64_t bp, int64_t stride, int32_t inverse);

typedef struct MinimalPlan MinimalPlan;

void do_fft_planned(MinimalPlan *P, MDArray *oy, MDArray *ix, int64_t r);

void do_fft(MDArray *oy, MDArray *ix, fft_func_t fn_name, int64_t *Ns,
            int64_t ndims, int32_t e1, int64_t r, int64_t bp, int64_t stride,
            int32_t inverse);

void fftr2(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr3(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr4(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr5(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr7(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr8(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void fftr9(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
           int64_t stride, int32_t inverse);
void direct_dft(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
                int64_t stride, int32_t inverse);
void bluestein(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t e1, int64_t bp,
               int64_t stride, int32_t inverse);

static fft_func_t dispatch[] = {NULL,   NULL, &fftr2, &fftr3, &fftr4,
                                &fftr5, NULL, &fftr7, &fftr8, &fftr9};

typedef void (*fft_func_2_t)(MFFTELEM **Y, MFFTELEM **X, int32_t e1, int32_t e2,
                             int64_t N1, int64_t N2, fft_func_t fft1,
                             fft_func_t fft2, int64_t bp, int64_t stride,
                             int32_t inverse);

void prime_factor_2(MFFTELEM **YY, MFFTELEM **XX, int32_t e1, int32_t e2,
                    int64_t N1, int64_t N2, fft_func_t fft1, fft_func_t fft2,
                    int64_t bp, int64_t stride, int32_t inverse);

typedef void (*fft_func_3_t)(MFFTELEM **Y, MFFTELEM **X, int32_t e1, int32_t e2,
                             int32_t e3, int64_t N1, int64_t N2, int64_t N3,
                             fft_func_t fft1, fft_func_t fft2, fft_func_t fft3,
                             int64_t bp, int64_t stride, int32_t inverse);

void prime_factor_3(MFFTELEM **YY, MFFTELEM **XX, int32_t e1, int32_t e2,
                    int32_t e3, int64_t N1, int64_t N2, int64_t N3,
                    fft_func_t fft1, fft_func_t fft2, fft_func_t fft3,
                    int64_t bp, int64_t stride, int32_t inverse);

static inline int64_t count_leading_zeros(uint64_t x) {
  return __builtin_clzll(x);
}

#endif // CMINIMALFFT_H