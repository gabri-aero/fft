#include <Eigen/Dense>
#include <complex>
#include "Fourier.hpp"
#include <random>

#include <gtest/gtest.h>

const int N = 8192;
Eigen::VectorX<double> y_random(N);
Eigen::VectorX<std::complex<double>> y_random_c(N);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);

// Test for add function
TEST(FourierTest, DFT)
{
    Eigen::VectorX<double> y(4);
    y << 0, 1, 4, 9;
    auto [c, s] = dft(y, 4);
    for (int i = 0; i < c.size(); i++)
    {
        std::cout << c[i] << ' ' << s[i] << std::endl;
    }
}

TEST(FourierTest, FFT)
{
    Eigen::VectorX<std::complex<double>> y(8);
    y << 0, 1, 4, 9, 10, 3, 4, 5;
    auto f = fft(y);
    std::cout << "FFT" << '\n';
    for (int i = 0; i < y.size(); i++)
    {
        std::cout << f[i] << std::endl;
    }
    fft(y_random_c);
}

TEST(FourierTest, RealFFT)
{
    constexpr int N = 16;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 2.0 * M_PI - (2.0 * M_PI / N));
    Eigen::VectorXd y = 3 + 2 * x.array().cos() - 4 * (4 * x.array()).sin();
    auto f = rfft(y);
    Eigen::VectorXd C(N), S(N);
    C(0) = f[0].real() / N; // C0 term
    S(0) = 0;
    for (int i = 1; i < f.size(); i++)
    {
        C(i) = 2 * f(i).real() / N;
        S(i) = -2 * f(i).imag() / N;
    }
    for (int i = 0; i < f.size(); i++)
    {
        std::cout << C(i) << " \t" << S(i) << "\n";
    }
    rfft(y_random);
}

TEST(FourierTest, Radix3)
{
    Eigen::VectorX<std::complex<double>> y(3);
    y << 0, 1, 4;
    auto f = fft_radix_3(y);
    ASSERT_NEAR(f(0).real(), 5, 1e-6);
    ASSERT_NEAR(f(0).imag(), 0, 1e-6);
    ASSERT_NEAR(f(1).real(), -2.5, 1e-6);
    ASSERT_NEAR(f(1).imag(), 2.59807621, 1e-6);
    ASSERT_NEAR(f(2).real(), -2.5, 1e-6);
    ASSERT_NEAR(f(2).imag(), -2.59807621, 1e-6);
}

TEST(FourierTest, Radix5)
{
    Eigen::VectorX<std::complex<double>> y(5);
    y << 0, 1, 4, 5, std::complex<double>(3, -2);
    auto f = fft_radix_5(y);
    ASSERT_NEAR(f(0).real(), 13, 1e-6);
    ASSERT_NEAR(f(0).imag(), -2, 1e-6);
    ASSERT_NEAR(f(1).real(), -4.14297194, 1e-6);
    ASSERT_NEAR(f(1).imag(), 1.8718643, 1e-6);
    ASSERT_NEAR(f(2).real(), 0.72065548, 1e-6);
    ASSERT_NEAR(f(2).imag(), 1.84254798, 1e-6);
    ASSERT_NEAR(f(3).real(), -1.63048553, 1e-6);
    ASSERT_NEAR(f(3).imag(), 1.39352, 1e-6);
    ASSERT_NEAR(f(4).real(), -7.947198, 1e-6);
    ASSERT_NEAR(f(4).imag(), -3.10793227, 1e-6);
}

TEST(FourierTest, CombinedRadixFFT)
{
    Eigen::VectorX<std::complex<double>> f(6);
    f << std::complex<double>(130.2738, 0), std::complex<double>(41.8896, 5.04759),
        std::complex<double>(1.945866, 10.136040), 115.2192,
        std::complex<double>(1.945866, -10.13604), std::complex<double>(41.8896, -5.04759);
    auto y = fft(f);
    ASSERT_NEAR(y(0).real(), 333.163932, 1e-6);
    ASSERT_NEAR(y(1).real(), 81.2971526, 1e-6);
    ASSERT_NEAR(y(2).real(), 192.84408007, 1e-6);
    ASSERT_NEAR(y(3).real(), -64.832868, 1e-6);
    ASSERT_NEAR(y(4).real(), 210.47098793, 1e-6);
    ASSERT_NEAR(y(5).real(), 28.6995154, 1e-6);
}

TEST(FourierTest, InverseFFT)
{
    Eigen::VectorX<std::complex<double>> f(6);
    f << std::complex<double>(130.2738, 0), std::complex<double>(41.8896, -5.04759),
        std::complex<double>(1.945866, -10.136040), 115.2192,
        std::complex<double>(1.945866, 10.13604), std::complex<double>(41.8896, 5.04759);
    auto y = ifft(f);
    ASSERT_NEAR(y(0).real(), 55.527322, 1e-6);
    ASSERT_NEAR(y(1).real(), 13.54952543, 1e-6);
    ASSERT_NEAR(y(2).real(), 32.14068001, 1e-6);
    ASSERT_NEAR(y(3).real(), -10.805478, 1e-6);
    ASSERT_NEAR(y(4).real(), 35.07849799, 1e-6);
    ASSERT_NEAR(y(5).real(), 4.78325257, 1e-6);
}