
# CMinimalFFT

This is a simple FFT library rewritten in C currently
supporting only the complex double type.  It is being used to study the performance compared to MinimalFFT.jl.

The Julia openlibm library is being used due to
providing a significantly faster sincos (sine and cosine) function compared with the standard C library.

## Organization

| Function | |
|---------------------|-------------------------------------------|
| Lowest level functions | fftr2!, fftr3!, direct_dft!, fft_rader!, etc. |
| Mid level decomposition functions | prime_factor!, mixed_radix! |
| Indexer functions | do_1d, do_1d_r1 |
| Multi-dimensional FFT indexers | do_fft_planned, do_1d |
| Planning functions | execute_plan |

## Testing

Currently only some simple testing is provided in test/test17.c and more testing will likely find bugs.  I have only tested this with test17.c on macOS.

## License

The license is MIT as described in LICENSE.txt


