// For approx N<20, the direct DFT can be as fast as the FFT
// due to lower communication cost. Also useful for testing.

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "cminimalfft.h"

typedef struct {
    complex double B;
    int64_t N;
    int32_t inverse;
} direct_buffer_t;

// Global buffer for caching twiddle factor
static direct_buffer_t direct_buff = {0.0 + 0.0*I, 0, 0};

// Direct DFT implementation
void direct_dft(MFFTELEM **Y, MFFTELEM **X, int64_t N, int32_t e1, int64_t bp, int64_t stride, int32_t inverse) {
    MFFTELEM *restrict y = *Y;
    MFFTELEM *restrict x = *X;

    double complex B;
    
    // Check if we can reuse cached twiddle factor
    if (direct_buff.N == N) {
        B = direct_buff.B;
        if (inverse != direct_buff.inverse) {
            B = conj(B);
            direct_buff.B = B;
            direct_buff.inverse = inverse;
        }
    } else {
        // Compute new twiddle factor
        minsincos(inverse ? 2.0 * M_PI / N : -2.0 * M_PI / N, imag(&B), real(&B));

        direct_buff.B = B;
        direct_buff.N = N;
        direct_buff.inverse = inverse;
    }
    
    MFFTELEM W_step = 1.0 + 0.0*I;
    
    for (int64_t k = 0; k < N; k++) {
        MFFTELEM W = 1.0 + 0.0*I;
        MFFTELEM s = 0.0 + 0.0*I;
        
        for (int64_t n = 0; n < N; n++) {
            s = s + W * x[bp + stride * n];
            W = W * W_step;
        }
        
        y[bp + stride * k] = s;
        W_step = W_step * B;
    }
    
}