#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "plan.h"

int64_t ipow(int64_t base, int32_t exp) {
    int64_t result = 1;
    while (exp > 0) {
        result *= base;
        exp--;
    }
    return result;
}

int add_factor(factorization *r, int64_t base, int32_t exponent) {
    if (r->count == MAX_FACTORS) {
        r->count++; // Signal too many factors
        return 1;
    }
    r->base[r->count] = base;
    r->n[r->count] = ipow(base, exponent);
    r->exponent[r->count++] = exponent;
    return 0;
}

factorization *factorize(int64_t n) {
    if (n < 2) {
        minassert(0,"Input must be a positive integer greater than 1");
        return 0;
    }

    factorization *result=(factorization *)calloc(1,sizeof(factorization));

    // Handle small primes separately
    int64_t primes[] = {2, 3, 5};
    for (int64_t i = 0; i < 3; i++) {
        int64_t exp = 0;
        while (n % primes[i] == 0) {
            n /= primes[i];
            exp++;
        }
        if (exp > 0) {
            if (add_factor(result, primes[i], exp))
                return result;
        }
    }

    // Wheel increments (coprime to 30): 7,11,13,17,19,23,29,31 → step pattern
    int64_t steps[] = {4, 2, 4, 2, 4, 6, 2, 6}; 
    int64_t step_count = 8;

    int64_t p = 7; // first candidate after 2,3,5
    int64_t step_index = 0;

    while (p * p <= n) {
        int64_t exp = 0;
        while (n % p == 0) {
            n /= p;
            exp++;
        }
        if (exp > 0) {
            if (add_factor(result, p, exp))
                return result;
        }
        p += steps[step_index];
        step_index = (step_index + 1) % step_count;
    }

    // If n is still > 1, it is prime
    if (n > 1) {
        if (add_factor(result, n, 1))
            return result;
    }

    return result;
}
