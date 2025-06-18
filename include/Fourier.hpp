/**
 * @file Fourier.hpp
 *
 * @brief Header file for Fourier Transform algorithms.
 *
 * This file contains the declarations of various Fourier Transform algorithms
 * such as DFT, FFT, RFFT, inverse FFT.
 *
 * @author Gabriel Valles
 * @date 2025-02-17
 */
#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include <math.h>

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <type_traits>

// Function prototypes
template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_2(const Eigen::VectorX<std::complex<T>> &y);

template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_3(const Eigen::VectorX<std::complex<T>> &y);

template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_5(const Eigen::VectorX<std::complex<T>> &y);

/**
 * @brief Computes the Discrete Fourier Transform (DFT) of a given real input
 * signal.
 *
 * This function computes the DFT of a real input signal using a  O(N²)
 * algorithm. If a non-zero number of output frequencies is provided, the output
 * is pruned accordingly.
 *
 * @tparam T The type of value of each signal sample.
 * @tparam N The length of the data
 * @param x The input signal represented as an Eigen::Vector of N samples.
 * @param nf The number of frequency bins to compute (0 by default for all
 * frequencies).
 * @return A std::pair<Eigen::Vector<U, N>, Eigen::Vector<U, N>> representing the cosine
 * and sine components of the frequency spectrum.
 */
template <typename T>
auto dft(const Eigen::VectorX<T> &y, const int nf)
{
    using U = std::common_type_t<double, T>;
    // Compute vector size
    int N = y.size();
    // Pre-allocate sine and cosine terms
    Eigen::VectorX<U> s(N);
    Eigen::VectorX<U> c(N);
    // Define Fourier matrices for sine and cosine
    Eigen::MatrixX<U> Fs(N, N);
    Eigen::MatrixX<U> Fc(N, N);
    U x;
    // Populate Fourier matrices
    for (int i = 0; i < nf; i++)
    {
        x = (2 * M_PI * i) / N;
        for (int j = 0; j < N; j++)
        {
            Fc(i, j) = cos(j * x);
            Fs(i, j) = sin(j * x);
        }
    }
    // Compute coefficients
    c = Fc * y;
    s = Fs * y;
    // Apply scaling
    c = c * 2.0 / N;
    s = s * 2.0 / N;
    c(0) = c(0) / 2.0; // Correction for c0
    return std::make_pair(c, s);
}

/**
 * @brief This function computes the Fast Fourier Transform of a given complex
 * input signal
 *
 * This function computes the FFT by means of the Cooley-Turkey algorithm, i.e.
 * using a Radix-2 Decimation in Time (DIT). It consists of an O(N log₂(N))
 * algorithm.
 *
 * @note No support is yet provided for signals with a number of samples
 * different than 2^p with p an integer
 *
 * @tparam T The raw type of value of every signal sample
 * @tparam N The length of the data signal
 * @param y The complex input signal as an Eigen::Vector<std::complex<T>>
 * @return An Eigen::Vector<std::complex<T>, N> representing the frequency spectrum
 */
template <typename T>
Eigen::VectorX<std::complex<T>> fft(const Eigen::VectorX<std::complex<T>> &y)
{
    // Determine signal size
    int N = y.size();
    // Handle signal size
    if (N == 1)
    {
        return y;
    }
    else if (N % 5 == 0)
    {
        return fft_radix_5<T>(y);
    }
    else if (N % 3 == 0)
    {
        return fft_radix_3<T>(y);
    }
    else if (N % 2 == 0)
    {
        return fft_radix_2<T>(y);
    }
    else
    {
        throw std::runtime_error(
            "Length of signal cannot be decomposed into available FFT-radix algorithms");
    }
}

template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_2(const Eigen::VectorX<std::complex<T>> &y)
{
    // Determine signal size
    int N = y.size();
    // Define types
    using U = std::complex<T>;
    // Define split size
    int M = N / 2;
    // Pre-allocate even and odd terms
    Eigen::VectorX<U> y_even(M), y_odd(M);
    // Populate even and odd terms
    for (int i = 0; i < M; i++)
    {
        y_even[i] = y[2 * i];
        y_odd[i] = y[2 * i + 1];
    }
    // Compute FFT for even and odd terms
    auto fhat_even = fft(y_even);
    auto fhat_odd = fft(y_odd);
    // Pre-allocate output frequencies
    Eigen::VectorX<U> fhat(N);
    // Define fundamental frequency
    T alpha = -2 * M_PI / N; // fundamental frequency angle
    U wi;                    // loop fundamental frequency
    T unit = static_cast<T>(1.0);
    for (int i = 0; i < M; i++)
    {
        wi = std::polar(unit, alpha * i);
        // Formulas for Radix-2 DIT
        fhat[i] = fhat_even[i] + wi * fhat_odd[i];
        fhat[i + M] = fhat_even[i] - wi * fhat_odd[i];
    }
    return fhat;
}

template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_3(const Eigen::VectorX<std::complex<T>> &y)
{
    // Determine signal size
    int N = y.size();
    // Define types
    using U = std::complex<T>;
    // Define split size
    int M = N / 3;
    // Pre-allocate groups
    std::vector<Eigen::VectorX<U>> yg(3), fg(3);
    // Populate group terms
    for (int j = 0; j < 3; j++)
    {
        yg[j].resize(M);
        fg[j].resize(M);
        for (int i = 0; i < M; i++)
        {
            yg[j][i] = y[3 * i + j];
        }
        fg[j] = fft(yg[j]);
    }
    // Pre-allocate output frequencies
    Eigen::VectorX<U> fhat(N);
    fhat.setZero();
    // Define fundamental frequency
    T unit = static_cast<T>(1.0);
    std::complex<T> wj = std::polar(unit, (T)-2 * M_PI / N);
    std::complex<T> wi = std::polar(unit, (T)-2 * M_PI / 3);
    // Loop for Radix-3 algorithm
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                fhat[i + M * j] += pow(wi, k * j) * pow(wj, k * i) * fg[k][i];
            }
        }
    }
    return fhat;
}

template <typename T>
Eigen::VectorX<std::complex<T>> fft_radix_5(const Eigen::VectorX<std::complex<T>> &y)
{
    // Determine signal size
    int N = y.size();
    // Define types
    using U = std::complex<T>;
    // Define split size
    int M = N / 5;
    // Pre-allocate groups
    std::vector<Eigen::VectorX<U>> yg(5), fg(5);
    // Populate group terms
    for (int j = 0; j < 5; j++)
    {
        yg[j].resize(M);
        fg[j].resize(M);
        for (int i = 0; i < M; i++)
        {
            yg[j][i] = y[5 * i + j];
        }
        fg[j] = fft(yg[j]);
    }
    // Pre-allocate output frequencies
    Eigen::VectorX<U> fhat(N);
    fhat.setZero();
    // Define fundamental frequency
    T unit = static_cast<T>(1.0);
    std::complex<T> wj = std::polar(unit, (T)-2 * M_PI / N);
    std::complex<T> wi = std::polar(unit, (T)-2 * M_PI / 5);
    // Loop for Radix-5 algorithm
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                fhat[i + M * j] += pow(wi, k * j) * pow(wj, k * i) * fg[k][i];
            }
        }
    }
    return fhat;
}

/**
 * @brief This function computes the Fast Fourier Transform of a given real
 * input signal
 *
 * This function leverages the frequency spectrum symmetry computing the FFT of
 * a complex signal with half the length of the input real signal. As a result
 * is 2x faster than the complex FFT.
 *
 * @note No support is yet provided for signals with a number of samples
 * different than 2^p with p an integer
 *
 * @tparam T The type of value of every signal sample
 * @tparam N The length of the data signal
 * @param y The complex input signal as a std::vector
 * @return An Eigen::Vector<std::complex<T>> representing the frequency spectrum
 */
template <typename T>
auto rfft(const Eigen::VectorX<T> &y)
{
    // Compute size
    const int N = y.size();
    // Define types
    using U = std::complex<T>;
    // Compute split size
    const int M = N / 2;
    // Build half-sized complex signal real input signal
    Eigen::VectorX<U> z(M); // Pre-allocate complex signal
    for (int i = 0; i < M; i++)
    {
        z[i] = U(y[2 * i], y[2 * i + 1]); // Even => real, Odd => imaginary
    }
    // Compute FFT of the complex signal
    auto fhat_z = fft(z);
    // Pre-allocate conjugated part
    Eigen::VectorX<U> fhat_z_conj(M);
    // Handle 0th term
    fhat_z_conj[0] = std::conj(fhat_z[0]);
    // Populate conjugated frequency spectrum
    for (int i = 1; i < M; i++)
    {
        fhat_z_conj[i] = std::conj(fhat_z[M - i]);
    }
    // Define imaginary unit
    U j(static_cast<T>(0), static_cast<T>(1));
    auto fhat_even = (fhat_z + fhat_z_conj) / 2;
    auto fhat_odd = -j * (fhat_z - fhat_z_conj) / 2;
    // Pre-allocate output frequencies
    Eigen::VectorX<U> fhat(N);
    // Define fundamental frequency
    T alpha = -2 * M_PI / N; // fundamental frequency angle
    U wi;                    // loop fundamental frequency
    T unit = static_cast<T>(1.0);
    for (int i = 0; i < M; i++)
    {
        wi = std::polar(unit, alpha * i);
        // Formula for Radix-2 DIT
        fhat[i] = fhat_even[i] + wi * fhat_odd[i];
        fhat[i + M] = fhat_even[i] - wi * fhat_odd[i];
    }
    return fhat;
}

template <typename T>
Eigen::VectorX<std::complex<T>> ifft(const Eigen::VectorX<std::complex<T>> &y)
{
    // Define type
    using U = std::complex<T>;
    // Compute size
    const int N = y.size();
    // Define conjugate
    Eigen::VectorX<U> yc(N);
    std::transform(y.begin(), y.end(), yc.begin(),
                   [](const std::complex<T> &x)
                   { return std::conj(x); });
    // Compute FFT of conjugate
    auto fc = fft(yc);
    // Compute conjugate of FFT result
    Eigen::VectorX<U> f(N);
    std::transform(fc.begin(), fc.end(), f.begin(),
                   [](const std::complex<T> &x)
                   { return std::conj(x); });
    return f / N;
}

#endif //_FOURIER_HPP_