
#include "MT_random.h"
#include "cminimalfft.h"
#include "hashmap.h"
#include "plan.h"
#include <fftw3.h>
#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define NUM_TIMED_TESTS 20
#define MAX_FACTORS 3

static fftw_plan create_fftw_plan(int n, MFFTELEM *in, MFFTELEM *out,
                                  int inverse) {
  if (inverse)
    return fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
                            FFTW_BACKWARD, FFTW_ESTIMATE);
  else
    return fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
                            FFTW_FORWARD, FFTW_ESTIMATE);
}

int64_t power_of(int64_t b, int64_t N) {
  int64_t count = 0;
  if (N <= 0)
    return 0;
  while (N % b == 0) {
    N /= b;
    count++;
    if (N == 1)
      return count;
  }
  return 0;
}

MFFTELEM *get_rv(rng_gaussian_type *RNG_p, size_t n) {
  MFFTELEM *arr = minaligned_alloc(sizeof(MFFTELEM), sizeof(MFFTELEM), n);
  for (int i = 0; i < n; ++i) {
    arr[i] = next_random_gaussian(RNG_p) + I * next_random_gaussian(RNG_p);
  }
  return arr;
}

void print_v(const char *name, MFFTELEM *v, size_t n) {
  printf("%s = \n", name);
  for (size_t i = 0; i < n; ++i) {
    printf("  (%2.2f,%2.2f)\n", creal(v[i]), cimag(v[i]));
  }
  printf("\n");
}

double get_s_time(int64_t start, int64_t end) {
  int64_t interval = end - start;
  double elapsed_ns = (double)interval;

  return elapsed_ns * 1e-9;
}

double get_bm_time(double *times, int n) {
  double t_min = DBL_MAX, t_max = 0.0;
  int i_min = -1, i_max = -1;
  for (int i = 0; i < n; ++i) {
    if ((times[i] != 0) && (times[i] < t_min)) {
      t_min = times[i];
      i_min = i;
    }
    if ((times[i] != 0) && (times[i] > t_max)) {
      t_max = times[i];
      i_max = i;
    }
  }
  return t_min;
}

void test_fft_kernel(int repeat_count, MFFTELEM **Y_ref, MFFTELEM **Y,
                     MFFTELEM **X_ref, MFFTELEM **X, MFFTELEM **copy_X,
                     fftw_plan P_ref, MinimalPlan *P, int64_t N, int bm,
                     double *t_ref_s, double *t_s, int32_t num_factors,
                     int64_t *Ns, fft_func_t *fns, int32_t *es, void *parent_fn,
                     int32_t inverse) {
  int64_t t_ref_start, t_ref_end;
  int64_t t_start, t_end;

  *t_ref_s = 0.0;
  *t_s = 0.0;

  memcpy(*X, *copy_X, N * sizeof(MFFTELEM));
  memcpy(*X_ref, *copy_X, N * sizeof(MFFTELEM));

  t_ref_start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  fftw_execute(P_ref);
  t_ref_end = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  if ((t_ref_end - t_ref_start) < 10000)
    repeat_count *= 40; // oversample if less than 10 us

  memcpy(*X_ref, *copy_X, N * sizeof(MFFTELEM));

  int num_tests = repeat_count;

  t_ref_start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  while (repeat_count-- > 0) {
    fftw_execute(P_ref);
    memcpy(*X_ref, *copy_X, N * sizeof(MFFTELEM));
  }
  t_ref_end = clock_gettime_nsec_np(CLOCK_MONOTONIC);

  repeat_count = num_tests;

  t_start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  while (repeat_count-- > 0) {
    if (P != NULL) {
      execute_plan(P, Y, X, 0, 0, 1);
      memcpy(*X, *copy_X, N * sizeof(MFFTELEM));

    } else {
      if (num_factors == 1) {
        ((fft_func_t)fns[0])(Y, X, N, es[0], 0, 1, inverse);
      } else if (num_factors == 2) {
        ((fft_func_2_t)parent_fn)(Y, X, es[0], es[1], Ns[0], Ns[1], fns[0],
                                  fns[1], 0, 1, inverse);
      } else if (num_factors == 3) {
        ((fft_func_3_t)parent_fn)(Y, X, es[0], es[1], es[2], Ns[0], Ns[1],
                                  Ns[2], fns[0], fns[1], fns[2], 0, 1, inverse);
      } else {
        bluestein(Y, X, N, 1, 0, 1, inverse);
      }
      memcpy(*X, *copy_X, N * sizeof(MFFTELEM));
    }
  }
  t_end = clock_gettime_nsec_np(CLOCK_MONOTONIC);

  if (bm) {
    *t_ref_s = get_s_time(t_ref_start, t_ref_end) / (double)num_tests;
    *t_s = get_s_time(t_start, t_end) / (double)num_tests;
  }
}

void print_fns(char *buf, fft_func_t *fns) {
  char fn[80];

  if (fns == NULL)
    return;

  for (int i = 0; i < MAX_FACTORS; ++i) {
    if (fns[i] == NULL)
      continue;
    for (int j = 0; j < sizeof(dispatch) / sizeof(dispatch[0]); ++j) {
      if (((fft_func_t)(fns[i])) == dispatch[j]) {
        sprintf(fn, "fftr%d ", j);
        strcat(buf, fn);
      }
    }
    if (fns[i] == (fft_func_t)bluestein) {
      sprintf(fn, "bluestein ");
      strcat(buf, fn);
    } else if (fns[i] == (fft_func_t)direct_dft) {
      sprintf(fn, "direct_dft ");
      strcat(buf, fn);
    }
  }
  if (strlen(buf))
    buf[strlen(buf) - 1] = 0; // remove last space
}

void print_result(const char *preamble, const char *name, int64_t N,
                  int num_factors, int64_t *Ns, int bm, double t_ref_s,
                  double t_s, fft_func_t *fns) {
  char timing[256];
  if (bm) {
    sprintf(timing, "time = %2.2e factor_ref = %2.2e", t_s, t_s / t_ref_s);
  } else {
    sprintf(timing, "untimed");
  }

  char fn_str[256];
  fn_str[0] = '\0';
  print_fns(fn_str, fns);

  switch (num_factors) {
  case 1:
    printf("%s %s N=%lld %s (%s)\n", preamble, name, Ns[0], timing, fn_str);
    break;
  case 2:
    printf("%s %s N=%lld=[%lld,%lld] %s (%s)\n", preamble, name, N, Ns[0],
           Ns[1], timing, fn_str);
    break;
  case 3:
    printf("%s %s N=%lld factors=%lld,%lld,%lld %s (%s)\n", preamble, name, N,
           Ns[0], Ns[1], Ns[2], timing, fn_str);
    break;
  default:
    printf("%s %s N=%lld %s (%s)\n", preamble, name, N, timing, fn_str);
    break;
  }
  fflush(stdout);
}

void test_fft(rng_gaussian_type *RNG, const char *name, int bm, int inverse,
              int64_t N, int *pc, int *fc, int num_factors, void *parent_fn,
              MinimalPlan *P, int64_t *Ns, fft_func_t *fns, int32_t *es) {

  struct timespec t_ref_start, t_ref_end;
  struct timespec t_start, t_end;
  double t_ref_s = DBL_MAX, t_s = DBL_MAX;

  MFFTELEM *Y_ref = minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N);
  MFFTELEM *X = minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N);
  MFFTELEM *Y = minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N);
  MFFTELEM *X_ref = get_rv(RNG, N);
  MFFTELEM *copy_X =
      memcpy(minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N), X_ref,
             N * sizeof(MFFTELEM));

  fftw_plan P_ref = create_fftw_plan(N, X_ref, Y_ref, inverse);

  int test_repeat = bm ? NUM_TIMED_TESTS : 1;

  test_fft_kernel(test_repeat, &Y_ref, &Y, &X_ref, &X, &copy_X, P_ref, P, N, bm,
                  &t_ref_s, &t_s, num_factors, Ns, fns, es, parent_fn, inverse);

  // reporting
  if (approx_cmp_v(Y_ref, Y, N)) {
    print_result("Failed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns);
    (*fc)++;
  } else {
    (*pc)++;
    print_result("Passed for", name, N, num_factors, Ns, bm, t_ref_s, t_s, fns);
  }
  free(X_ref);
  free(X);
  free(copy_X);
  free(Y_ref);
  free(Y);
}

static const int64_t factor_1[][1] = {
    {8}, {4}, {25}, {27}, {16}, {125}, {49}, {64}, {81}, {9 * 9 * 9}, {256}};

static const int64_t factor_2[][2] = {
    {4, 25},   {25, 4}, {4, 49}, {8, 9},  {256, 25}, {25, 256},
    {16, 5},   {8, 7},  {11, 8}, {25, 4}, {49, 3},   {9, 8},
    {25, 256}, {16, 5}, {8, 7},  {25, 4}, {1, 256}};

static const int64_t factor_3[][3] = {
    {9, 5, 49},    {9, 49, 5},    {5, 9, 49},   {49, 5, 9}, {8, 7, 25},
    {7, 25, 8},    {2, 3, 5},     {2, 5, 3},    {3, 2, 5},  {3, 5, 2},
    {64, 3, 5},    {3, 5, 64},    {5, 64, 3},   {1, 1, 64}, {27, 625, 49},
    {625, 27, 49}, {49, 27, 625}, {49, 625, 27}};

static const int64_t high_radix_factor_1[][1] = {
    {262144}, {2097152}, // powers of 8
    {531441}, {4782969},  // powers of 9
    {78125},  {390625},  // powers of 5
    {823543}, {5764801}  // powers of 7
};

void hex_dump(const void *ptr, size_t len) {
  const unsigned char *data = (const unsigned char *)ptr;
  for (size_t i = 0; i < len; ++i) {
    printf("%02X ", data[i]);
    if ((i + 1) % 16 == 0)
      printf("\n");
  }
  if (len % 16 != 0)
    printf("\n");
}

int get_key_length() {
  int total_len = 1;
  total_len += sizeof(void *);
  total_len += sizeof(void *) * MAX_FACTORS;
  total_len += sizeof(int64_t) * MAX_FACTORS;
  total_len += sizeof(int32_t) * MAX_FACTORS;
  total_len += sizeof(char);
  total_len += sizeof(int32_t);
  return total_len;
}

char *write_to_key(char *s, void *data, size_t len) {
  memcpy(s, data, len);
  s += len;
  return s;
}

char *getkey(int key_length, void *parent_fn, void *fns, int64_t *N_vals,
             int32_t *es, char bm, int32_t inverse) {
  char *orig_s = calloc(key_length, sizeof(char));
  char *s = orig_s;
  s = write_to_key(s, parent_fn, sizeof(void *));
  s = write_to_key(s, fns, sizeof(void *) * MAX_FACTORS);
  s = write_to_key(s, N_vals, sizeof(int64_t) * MAX_FACTORS);
  s = write_to_key(s, es, sizeof(int32_t) * MAX_FACTORS);
  s = write_to_key(s, &bm, sizeof(char));
  s = write_to_key(s, &inverse, sizeof(inverse));
  return orig_s;
}

int64_t prod(int64_t *arr, int len) {
  int64_t p = 1;
  for (int i = 0; i < len; ++i) {
    p *= arr[i];
  }
  return p;
}

void driver(rng_gaussian_type *RNG_p, struct hashmap_s *d, int *radix,
            int radix_count, int num_factors, int bm, int *pc, int *fc,
            int32_t inverse, void *parent_fn, const int64_t *N_vals,
            int N_val_count, const char *name) {

  const int key_len = get_key_length();
  int64_t bs[MAX_FACTORS];
  int32_t es[MAX_FACTORS];
  fft_func_t fns[MAX_FACTORS];
  int64_t Ns[MAX_FACTORS];

  for (int t = 0; t < N_val_count; t++) {
    int64_t i = 0;
    memset(fns, 0, MAX_FACTORS * sizeof(fft_func_t));
    memset(Ns, 0, MAX_FACTORS * sizeof(int64_t));
    memset(es, 0, MAX_FACTORS * sizeof(int32_t));
    for (int f = 0; f < num_factors; f++) {
      const int64_t factor = N_vals[t * num_factors + f];
      Ns[f] = factor;
      for (int j = 0; j < radix_count; j++) {
        int r = radix[j];
        minassert(r < sizeof(dispatch) / sizeof(dispatch[0]), "Invalid radix.");
        int64_t e = power_of(r, factor);
        if (e != 0) {
          es[i] = e;
          bs[i] = r;
          fns[i] = dispatch[r];
          minassert(fns[i], "Function not available.");
          i++;
          break;
        }
      }
      char *key = getkey(key_len, parent_fn, fns, Ns, es, bm, inverse);
      void *element = hashmap_get(d, key, key_len);
      if (element != NULL || i != num_factors)
        continue;
      hashmap_put(d, key, key_len, (void *)1);
      test_fft(RNG_p, name, bm, inverse, prod(Ns, num_factors), pc, fc,
               num_factors, parent_fn, 0, Ns, fns, es);
    }
  }
}

void print_time() {
  time_t now = time(NULL);
  struct tm *t = localtime(&now);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
  printf("# %s\n\n", buf);
}

void get_plan_fns(fft_func_t *fns, int r, MinimalPlan *P) {
  for (int i = 0; i < MAX_FACTORS; ++i)
    fns[i] = NULL;

  int limit = P->num_factors[r];
  for (int i = 0; i < limit; ++i) {
    fns[i] = P->ip[r][i].func;
  }
}

int main() {
  const unsigned initial_size = 64;
  struct hashmap_s d;
  rng_gaussian_type RNG = init_rng_gaussian(6502, 0.0, 1.0);

  int code = hashmap_create(initial_size, &d);
  assert(code == 0 && "Failed to create hashmap.");

  int pass = 0, fail = 0;
  int *pc = &pass;
  int *fc = &fail;

  print_time();

#define RUN_DRIVER(radix_arr, num_factors, bm, inverse, parent_fn, N_vals,     \
                   name)                                                       \
  driver(&RNG, &d, (radix_arr), sizeof(radix_arr) / sizeof((radix_arr)[0]),    \
         (num_factors), 1 /*(bm)*/, pc, fc, (inverse), (parent_fn),            \
         &(N_vals)[0][0], sizeof((N_vals)) / sizeof(N_vals[0]), (name))

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, 0, 0, abort, factor_1,
             "stockham test 6");

  RUN_DRIVER(((int[]){2}), 1, 1, 0, abort, factor_1, "timed stockham test 0");
  RUN_DRIVER(((int[]){3}), 1, 1, 0, abort, factor_1, "timed stockham test 1");
  RUN_DRIVER(((int[]){4}), 1, 1, 0, abort, factor_1, "timed stockham test 2");
  RUN_DRIVER(((int[]){5}), 1, 1, 0, abort, factor_1, "timed stockham test 3");
  RUN_DRIVER(((int[]){7}), 1, 1, 0, abort, factor_1, "timed stockham test 4");
  RUN_DRIVER(((int[]){8}), 1, 1, 0, abort, factor_1, "timed stockham test 5");
  RUN_DRIVER(((int[]){9}), 1, 1, 0, abort, factor_1, "timed stockham test 6");

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 0");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 1");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 2");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 3");
  RUN_DRIVER(((int[]){4, 9, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 4");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 5");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 1, 0, 1, abort, factor_1,
             "stockham inverse test 6");

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 0, 1, prime_factor_2, factor_2,
             "prime factor 2 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 2, 1, 0, prime_factor_2, factor_2,
             "prime factor 2 test 1 timed");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 2, 0, 0, prime_factor_2, factor_2,
             "prime factor 2 test 5");

  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 0, 1, prime_factor_3, factor_3,
             "prime factor 3 test 0 inverse");
  RUN_DRIVER(((int[]){2, 3, 5, 7}), 3, 1, 0, prime_factor_3, factor_3,
             "prime factor 3 test 1");
  RUN_DRIVER(((int[]){4, 3, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 2");
  RUN_DRIVER(((int[]){8, 3, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 3");
  RUN_DRIVER(((int[]){2, 9, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 4");
  RUN_DRIVER(((int[]){8, 9, 5, 7}), 3, 0, 0, prime_factor_3, factor_3,
             "prime factor 3 test 5");

  hashmap_destroy(&d);
  code = hashmap_create(initial_size, &d);
  assert(code == 0 && "Failed to create hashmap 2.");

  RUN_DRIVER(((int[]){3}), 1, 0, 0, abort, high_radix_factor_1, "radix 3 test");

  RUN_DRIVER(((int[]){4}), 1, 0, 0, abort, high_radix_factor_1, "radix 4 test");

  RUN_DRIVER(((int[]){5}), 1, 0, 0, abort, high_radix_factor_1, "radix 5 test");

  RUN_DRIVER(((int[]){7}), 1, 0, 0, abort, high_radix_factor_1, "radix 7 test");

  RUN_DRIVER(((int[]){8}), 1, 0, 0, abort, high_radix_factor_1, "radix 8 test");

  RUN_DRIVER(((int[]){9}), 1, 0, 0, abort, high_radix_factor_1, "radix 9 test");


  static int64_t planner_n[] = {
      100,     196,    72,  6400,   80,
      56,      100,    147, 72,     2205,
      1400,    30,     960, 826875, 2 * 3 * 5 * 7 * 11 * 13,
      1 << 20, 1 << 22};

  int64_t *planner_n_inverse = planner_n;

  for (int i = 0; i < sizeof(planner_n) / sizeof(planner_n[0]); ++i) {
    int64_t N = planner_n[i];
    int64_t *N_p = &N;

    MinimalPlan *P = create_min_plan(N_p, 1, 0, 0, P_NONE);
    fft_func_t fns[MAX_FACTORS];
    get_plan_fns(fns, 0, P);

    test_fft(&RNG, "planner", 1, 0, N, pc, fc, 0, NULL, P, &N, fns, NULL);
    free_min_plan(P);

    MinimalPlan *P_inv = create_min_plan(N_p, 1, 0, 0, P_INVERSE);

    get_plan_fns(fns, 0, P_inv);
    test_fft(&RNG, "planner inverse", 1, 1, N, pc, fc, 0, NULL, P_inv, &N, fns,
             NULL);
    free_min_plan(P_inv);
  }

  printf("\nPassed %d tests.\n", pass);
  printf("Failed %d tests.\n", fail);
  fflush(stdout);
  return 0;
}