// Bluestein's FFT algorithm

#include "cminimalfft.h"
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  MFFTELEM *restrict a_n;
  MFFTELEM *restrict b_n;
  MFFTELEM *restrict A_X;
  MFFTELEM *restrict B_X;
  int64_t M;
  int32_t inverse;
} bs_buffer_t;

// Global buffer for Bluestein algorithm (no dictionary needed)
static bs_buffer_t bs_buff = {NULL, NULL, NULL, NULL, 0, 0};

static inline int64_t nextpow2_exp(uint64_t n) {
  if (n == 0)
    return 0;
  if (n == 1)
    return 2;

  // Find the position of the highest set bit
  int64_t highest_bit = count_leading_zeros(n);

  return highest_bit + 1; // increment even if already power of 2
}

// Bluestein FFT implementation
void bluestein(MFFTELEM **YY, MFFTELEM **XX, int64_t N, int32_t discard_e1, int64_t bp, int64_t stride, int32_t inverse) {
  MFFTELEM *restrict y = *YY;
  MFFTELEM *restrict x = *XX;

  int32_t e1 = nextpow2_exp(2 * N - 1);
  int64_t M = 1 << e1;

  int64_t init = bs_buff.M == 0;

  if (!init) {
    if (bs_buff.M != M) {
      init = 1;
    } else if (bs_buff.inverse != inverse) {
      // Conjugate b_n array
      for (int64_t i = 0; i < M; i++) {
        bs_buff.b_n[i] = conj(bs_buff.b_n[i]);
      }
      bs_buff.inverse = inverse;
    }
  }

  if (init) {
    // Free existing buffers if they exist
    if (bs_buff.b_n != NULL) {
      free(bs_buff.a_n);
      free(bs_buff.b_n);
      free(bs_buff.A_X);
      free(bs_buff.B_X);
    }

    // Allocate new buffers
    bs_buff.a_n = (MFFTELEM *restrict)malloc(M * sizeof(MFFTELEM));
    bs_buff.b_n = (MFFTELEM *restrict)calloc(M, sizeof(MFFTELEM));
    bs_buff.A_X = (MFFTELEM *restrict)malloc(M * sizeof(MFFTELEM));
    bs_buff.B_X = (MFFTELEM *restrict)malloc(M * sizeof(MFFTELEM));

    // Initialize b_n
    bs_buff.b_n[0] = 1.0 + 0.0 * I;
    for (int64_t n = 1; n < N; n++) {
      double arg = inverse ? -M_PI / N : M_PI / N;
      double complex c_e;
      minsincos(arg * n * n, imag(&c_e), real(&c_e));
      bs_buff.b_n[n] = c_e;
      bs_buff.b_n[M-n] = c_e;
    }

    bs_buff.M = M;
    bs_buff.inverse = inverse;
  }

  // TBD: add restrict while passing to MFFTELEM **
  MFFTELEM * a_n = bs_buff.a_n;
  const MFFTELEM * b_n = bs_buff.b_n;
  MFFTELEM * A_X = bs_buff.A_X;
  MFFTELEM * B_X = bs_buff.B_X;

  // Zero out a_n
  memset(a_n, 0, M * sizeof(MFFTELEM));

  // Fill a_n and y arrays
  for (int64_t n = 0; n < N; n++) {
    MFFTELEM c = conj(b_n[n]);
    a_n[n] = x[bp + stride * n] * c;
    y[bp + stride * n] = c;
  }

  // Copy b_n to B_X
  memcpy(B_X, b_n, M * sizeof(MFFTELEM));

  // Perform convolution using FFT
  fftr2(&A_X, &a_n, M, e1, 0, 1, 0);
  fftr2(&a_n, &B_X, M, e1, 0, 1, 0); // Forward FFT of b_n

  // Element-wise multiplication
  for (int64_t i = 0; i < M; i++) {
    a_n[i] *= A_X[i];
  }

  // Inverse FFT
  fftr2(&B_X, &a_n, M, e1, 0, 1, 1);

  // Scale by 1/M
  double scale = 1.0 / M;
  for (int64_t i = 0; i < M; i++) {
    B_X[i] *= scale;
  }

  // Final multiplication
  for (int64_t i = 0; i < N; i++) {
    y[bp + stride * i] *= B_X[i];
  }
}

// Cleanup function to free global buffer
void free_bluestein_buffer(void) {
  if (bs_buff.M != 0) {
    free(bs_buff.a_n);
    free(bs_buff.b_n);
    free(bs_buff.A_X);
    free(bs_buff.B_X);
    bs_buff.M = 0;
  }
}