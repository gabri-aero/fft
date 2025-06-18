# FFT

![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gabri-aero/fft/blob/main/LICENSE)

> A header-only library that provides Fast Fourier Transform utilities.

## Features
- Discrete Fourier Transform (DFT) for a defined number of frequencies.
- Fast Fourier Transform (FFT) employing radix-2,-3,-5 schemes.
- Inverse Fast Fourier Transform (iFFT).
- Real Fast Fourier Transform (RFFT), leveraging symmetries from real signals to halve computation time. 

> [!WARNING]
> No support is provided for signals with a number of samples that is not multiple of 2, 3 or 5.

## Building and Running Tests
 
To build and run the tests, simply run:

```bash
make tests
