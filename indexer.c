// indexer.c - column-major indexing

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cminimalfft.h"
#include "plan.h"

void compute_strides(int64_t *strides, int64_t *dims, int64_t ndims, int64_t instride) {
    int64_t prod = 1;
    for (int64_t s = 0; s < ndims; s++) {
        strides[s] = instride*prod;
        prod *= dims[s];
    }
}

int compute_product(int64_t *dims, int64_t ndims) {
    int64_t sz = 1;
    for (int64_t d = 0; d < ndims; d++) {
        sz *= dims[d];
    }
    return sz;
}

int64_t indexer_count(int64_t r, int64_t ndims, int64_t *counts, 
    int64_t *strides, int64_t bp, int64_t *dims) {
    int64_t i = 0;
    while (i < ndims) {
        if (i != r) {
            counts[i] += 1;
            if (counts[i] == dims[i]) {
                counts[i] = 0;
                bp -= strides[i] * (dims[i] - 1);
            } else {
                bp += strides[i];
                break;
            }
        }
        i++;
    }
    return (i==ndims) ? -1 : bp;
}

void do_1d_plan(MinimalPlan *P, MDArray *oy, MDArray *ix, int64_t r) {
    int64_t ndims = oy->ndims;
    int64_t *strides = (int64_t *)malloc((uint64_t)ndims * sizeof(int64_t));
    int64_t *counts = (int64_t *)calloc((uint64_t)ndims, sizeof(int64_t));
    
    compute_strides(strides, oy->dims, ndims, 1);
    
    int64_t bp = 0;
    int64_t flipped = 0;

    MFFTELEM **y = &(oy->data);
    MFFTELEM **x = &(ix->data);
    MFFTELEM *orig_y = *y;
    
    int64_t stride = strides[r];
    
    while (bp != -1) {
        execute_plan(P, y, x, r, bp, stride);
        
        if (*y != orig_y) {
            flipped = 1;
            swap_ptrs(y, x);
        }
        bp = indexer_count(r, ndims, counts, strides, bp, oy->dims);
    }
    
    if (flipped) {
        swap_ptrs(y, x);
    }
    
    free(strides);
    free(counts);
}

void do_1d_r0(MinimalPlan *P, MDArray *oy, MDArray *ix) {
    int64_t vlength = oy->dims[0];
    int64_t bp = 0;
    int64_t limit = oy->total_size;
    int64_t flipped = 0;

    MFFTELEM **y = &(oy->data);
    MFFTELEM **x = &(ix->data);
    MFFTELEM *orig_y = *y;
    
    while (bp < limit) {
        execute_plan(P, y, x, 0, bp, 1);
        
        if (*y != orig_y) {
            flipped = 1;
            swap_ptrs(y, x);
        }
        bp += vlength;
    }
    
    if (flipped) {
        swap_ptrs(y, x);
    }
}

// do_fft_planned function
void do_fft_planned(MinimalPlan *P, MDArray *oy, MDArray *ix, int64_t r) {
    if (r == 0) {
        do_1d_r0(P, oy, ix);
    } else {
        do_1d_plan(P, oy, ix, r);
    }
}
// do_1d function without plan
void do_1d_func(MDArray *oy, MDArray *ix, fft_func_t fn_name, int64_t *Ns, int64_t ndims,
                int32_t e1, int64_t r, int64_t bp, int64_t instride, int32_t inverse) {
    int64_t *strides = malloc((uint64_t)ndims * sizeof(int64_t));
    int64_t *counts = calloc((uint64_t)ndims, sizeof(int64_t));
    
    // Compute strides for embedded array
    compute_strides(strides, Ns, ndims, instride);
    
    int64_t vlength = Ns[r];
    int64_t flipped = 0;
    
    MFFTELEM **y = &(oy->data);
    MFFTELEM **x = &(ix->data);
    MFFTELEM *orig_y = *y;
    
    int64_t stride = strides[r];
    
    while (bp != -1) {
        fn_name(y, x, vlength, e1, bp, stride, inverse);
        
        if (*y != orig_y) {
            flipped = 1;
            swap_ptrs(y, x);
        }
        
        bp = indexer_count(r, ndims, counts, strides, bp, Ns);
    }
    
    if (flipped) {
        swap_ptrs(y, x);
    }
    
    free(strides);
    free(counts);
}

void do_1d_r0_func(MDArray *oy, MDArray *ix, fft_func_t fn_name, int64_t *Ns, int64_t ndims,
                   int32_t e1, int64_t bp, int64_t stride, int32_t inverse) {
    int64_t vlength = Ns[0];
    int64_t flipped = 0;
    
    MFFTELEM **y = &(oy->data);
    MFFTELEM **x = &(ix->data);
    MFFTELEM *orig_y = *y;
    
    // Compute limit from dimensions 1:end
    int64_t limit = compute_product(Ns+1, ndims-1);
    int64_t l = 0;
    
    while (l < limit) {
        l++;
        fn_name(y, x, vlength, e1, bp, stride, inverse);
        
        if (*y != orig_y) {
            flipped = 1;
            swap_ptrs(y, x);
        }
        bp += stride * vlength;
    }
    
    if (flipped) {
        swap_ptrs(y, x);
    }
}

// do_fft function
void do_fft(MDArray *oy, MDArray *ix, fft_func_t fn_name, int64_t *Ns, int64_t ndims,
            int32_t e1, int64_t r, int64_t bp, int64_t stride, int32_t inverse) {
    if (r == 0) {
        do_1d_r0_func(oy, ix, fn_name, Ns, ndims, e1, bp, stride, inverse);
    } else {
        do_1d_func(oy, ix, fn_name, Ns, ndims, e1, r, bp, stride, inverse);
    }
}