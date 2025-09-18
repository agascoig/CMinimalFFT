#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "cminimalfft.h"
#include <fftw3.h>

// Extended Euclidean algorithm
typedef struct {
    int64_t g, x, y;
} ExtendedEuclidResult;

ExtendedEuclidResult extended_euclid(int64_t a, int64_t b) {
    minassert(a >= 0 && b >= 0,"a and b must be non-negative");
    
    if (a == 0) {
        ExtendedEuclidResult result = {b, 0, 1};
        return result;
    }
    
    ExtendedEuclidResult sub_result = extended_euclid(b % a, a);
    ExtendedEuclidResult result = {
        sub_result.g,
        sub_result.y - (b / a) * sub_result.x,
        sub_result.x
    };
    return result;
}

// Inline mask mux mod function
static inline int64_t mask_mux_mod(int64_t a, int64_t B) {
    return a - (B & -(a >= B));
}

// Two-factor prime factor algorithm
void prime_factor_2(MFFTELEM **restrict YY, MFFTELEM **restrict XX, int32_t e1, int32_t e2, int64_t N1, int64_t N2,
                   fft_func_t fft1, fft_func_t fft2, int64_t bp, int64_t stride, int32_t inverse) {
    MFFTELEM *restrict Y = *YY;
    MFFTELEM *restrict X = *XX;
    
    int64_t N = N1 * N2;
    int64_t Ns[2] = {N1, N2};
    
    ExtendedEuclidResult result = extended_euclid(N1, N2);
    minassert(result.g == 1,"prime_factor N1 and N2 must be coprime");
    
    int64_t M1 = result.x;
    int64_t M2 = result.y;
    
    int64_t Q1P = M2 % N1;
    if (Q1P < 0) Q1P += N1;
    
    // Forward mapping
    int64_t rhs_n = 0;
    int64_t L2 = 0;
    for (int64_t n1p = 0; n1p < N1; n1p++) {
        int64_t R1 = 0;
        L2 = 0;
        for (int64_t n2p = 0; n2p < N2; n2p++) {
            int64_t n1 = mask_mux_mod(n1p + R1, N1);
            int64_t lhs_n = n1 + L2;
            Y[bp + stride * lhs_n] = X[bp + stride * rhs_n];
            R1 = mask_mux_mod(R1 + Q1P, N1);
            rhs_n++;
            L2 += N1;
        }
    }
    
    // Create 2D arrays for FFT operations
    MDArray Y2D = create_mdarray(Y, Ns, 2);
    MDArray X2D = create_mdarray(X, Ns, 2);
    
    // Perform FFTs
    do_fft(&X2D, &Y2D, fft1, Ns, 2, e1, 0, bp, stride, inverse);
    do_fft(&Y2D, &X2D, fft2, Ns, 2, e2, 1, bp, stride, inverse);

    // Rebind X and Y
    Y = Y2D.data;
    X = X2D.data;

    // Backward mapping
    int64_t Q2P = M1 % N2;
    if (Q2P < 0) Q2P += N2;
    
    int64_t lhs_k = 0;
    for (int64_t k2p = 0; k2p < N2; k2p++) {
        int64_t R1 = 0;
        for (int64_t k1p = 0; k1p < N1; k1p++) {
            int64_t k2 = mask_mux_mod(k2p + R1, N2);
            int64_t rhs_k = k1p + k2 * N1;
            X[bp + stride * lhs_k] = Y[bp + stride * rhs_k];
            R1 = mask_mux_mod(R1 + Q2P, N2);
            lhs_k++;
        }
    }
    *YY = X;
    *XX = Y;
}

// Structure for three-factor Q values
typedef struct {
    int64_t p1, p2, p3, p4;
    int64_t Q1, Q2, Q3, Q4;
} QValues;

QValues compute_Qs(int64_t N1, int64_t N2, int64_t N3) {
    ExtendedEuclidResult r1 = extended_euclid(N1, N2 * N3);
    ExtendedEuclidResult r2 = extended_euclid(N2, N1 * N3);
    ExtendedEuclidResult r3 = extended_euclid(N3, N1 * N2);
    ExtendedEuclidResult r4 = extended_euclid(N2 * N3, N1);
    
    minassert(r1.g == 1 && r2.g == 1 && r3.g == 1 && r4.g == 1,
           "N1, N2, N3 must be coprime");
    
    QValues q = {
        r1.x, r2.x, r3.x, r4.x,
        -r1.y, -r2.y * N1, -r3.y * N1, -r4.y
    };
    return q;
}

// Forward n-mapping for 3-factor
void nmap_3(MFFTELEM *restrict Y, MFFTELEM *restrict X, int64_t bp, int64_t stride, 
           int64_t N1, int64_t N2, int64_t N3, int64_t Q1P, int64_t Q2P) {
    int64_t rhs_n = 0;
    for (int64_t n1p = 0; n1p < N1; n1p++) {
        int64_t R1 = 0;
        for (int64_t n2p = 0; n2p < N2; n2p++) {
            int64_t R2 = 0;
            for (int64_t n3p = 0; n3p < N3; n3p++) {
                int64_t n1 = mask_mux_mod(n1p + R1, N1);
                int64_t n2 = mask_mux_mod(n2p + R2, N2);
                int64_t lhs_n = n1 + N1 * n2 + N1 * N2 * n3p;
                Y[bp + stride * lhs_n] = X[bp + stride * rhs_n];
                R1 = mask_mux_mod(R1 + Q1P, N1);
                R2 = mask_mux_mod(R2 + Q2P, N2);
                rhs_n++;
            }
        }
    }
}

// Backward k-mapping for 3-factor
void kmap_3(MFFTELEM *restrict Y, MFFTELEM *restrict X, int64_t bp, int64_t stride,
           int64_t N1, int64_t N2, int64_t N3, int64_t P1, int64_t P2) {
    int64_t lhs_k = 0;
    for (int64_t k3p = 0; k3p < N3; k3p++) {
        int64_t R2 = 0;
        for (int64_t k2p = 0; k2p < N2; k2p++) {
            int64_t R1 = 0;
            for (int64_t k1p = 0; k1p < N1; k1p++) {
                int64_t k2 = mask_mux_mod(k2p + R1, N2);
                int64_t k3 = mask_mux_mod(k3p + R2, N3);
                int64_t rhs_k = k1p + N1 * k2 + N1 * N2 * k3;
                Y[bp + stride * lhs_k] = X[bp + stride * rhs_k];
                R1 = mask_mux_mod(R1 + P1, N2);
                R2 = mask_mux_mod(R2 + P2, N3);
                lhs_k++;
            }
        }
    }
}

// Three-factor prime factor algorithm
void prime_factor_3(MFFTELEM **restrict YY, MFFTELEM **restrict XX, int32_t e1, int32_t e2, int32_t e3,
                   int64_t N1, int64_t N2, int64_t N3,
                   fft_func_t fft1, fft_func_t fft2, fft_func_t fft3,
                   int64_t bp, int64_t stride, int32_t inverse) {
    MFFTELEM *restrict Y = *YY;
    MFFTELEM *restrict X = *XX;
    
    int64_t N = N1 * N2 * N3;
    int64_t Ns[3] = {N1, N2, N3};
    
    QValues B = compute_Qs(N1, N2, N3);
    
    int64_t Q1P = (-B.Q1) % N1;
    if (Q1P < 0) Q1P += N1;
    
    int64_t Q2P = (-B.Q2) % N2;
    if (Q2P < 0) Q2P += N2;
    
    int64_t P1 = (-B.Q4) % N2;
    if (P1 < 0) P1 += N2;
    
    int64_t P2 = ((-B.Q3) / N1) % N3;
    if (P2 < 0) P2 += N3;
    
    // Forward mapping
    nmap_3(Y, X, bp, stride, N1, N2, N3, Q1P, Q2P);
    
    // Create 3D arrays for FFT operations
    MDArray Y123 = create_mdarray(Y, Ns, 3);
    MDArray X123 = create_mdarray(X, Ns, 3);
    
    // Perform FFTs
    do_fft(&X123, &Y123, fft1, Ns, 3, e1, 0, bp, stride, inverse);
    do_fft(&Y123, &X123, fft2, Ns, 3, e2, 1, bp, stride, inverse);
    do_fft(&X123, &Y123, fft3, Ns, 3, e3, 2, bp, stride, inverse);

    // Rebind
    Y = Y123.data;
    X = X123.data;

    // Backward mapping
    kmap_3(Y, X, bp, stride, N1, N2, N3, P1, P2);
    
    *YY = Y;
    *XX = X;
}