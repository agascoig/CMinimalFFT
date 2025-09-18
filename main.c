
#include "cminimalfft.h"
#include "random.h"
#include <stdio.h>
#include <time.h>

complex double testcomplex1(complex double z) { return times_pmim(z, 0); }

complex double testcomplex2(complex double z) { return times_pmim(z, 1); }

void vmfftprint(const MFFTELEM *arr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    complex double elem = arr[i];
    printf("(%f, %f)\n", creal(elem), cimag(elem));
  }
}

int main() {
  int32_t e1 = 22;
  int64_t N = 1 << e1;

  MDArray oy, ix;
  oy.data = minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N);
  oy.ndims = 1;
  oy.total_size = N;
  oy.dims[0] = N;

  ix.data = minaligned_calloc(sizeof(MFFTELEM), sizeof(MFFTELEM), N);
  ix.data[1] = 1.0 + 0.0 * I;
  ix.ndims = 1;
  ix.total_size = N;
  ix.dims[0] = N;

  fftr2(&oy.data, &ix.data, N, e1, 0, 1, 0);
  memset(oy.data, 0, N * sizeof(MFFTELEM));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  //    do_fft(&oy, &ix, fftr2, &N, 1, 1, 0, 0, 1, 0);

  fftr2(&oy.data, &ix.data, N, e1, 0, 1, 0);

  clock_gettime(CLOCK_MONOTONIC, &end);

  long long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL +
                         (end.tv_nsec - start.tv_nsec);

  // Convert to seconds for display
  double elapsed_seconds = elapsed_ns / 1000000000.0;

  printf("FFT took: %.6f seconds (%.0f nanoseconds)\n", elapsed_seconds,
         (double)elapsed_ns);

  free(oy.data);
  free(ix.data);
}