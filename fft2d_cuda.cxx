// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights
// reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <complex>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cassert>

#include <cuda_runtime.h>
#include <cufft.h>

int main()
{
    std::cout << "cufft 2D double-precision real-to-complex transform\n";

    const size_t Nx = 17;
    const size_t Ny = 17;
    const size_t Nycomplex = Ny / 2 + 1;

    std::cout << "Nx: " << Nx << "\tNy " << Ny
              << "\tNycomplex " << Nycomplex << std::endl;

    const size_t rstride   = Ny;

    const size_t real_bytes    = sizeof(double) * Nx * rstride;
    const size_t complex_bytes = 2 * sizeof(double) * Nx * Nycomplex;

    double *x, *x2;
    std::complex<double>* y;
    cudaError_t rt;
    rt = cudaMalloc((void **)&x, real_bytes);
    assert(rt == cudaSuccess);
    rt = cudaMalloc((void **)&x2, real_bytes);
    assert(rt == cudaSuccess);
    rt = cudaMalloc((void **)&y, complex_bytes);
    assert(rt == cudaSuccess);

    std::vector<double> rdata(Nx * rstride);
    // 2d delta function, zero except for one point that is 1.0
    rdata[Nx * 8 + 8] = 1.0;

    rt = cudaMemcpy(x, rdata.data(), real_bytes, cudaMemcpyHostToDevice);
    assert(rt == cudaSuccess);

    std::cout << "input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * rstride + j;
            std::cout << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Create plan:
    cufftResult rc   = CUFFT_SUCCESS;
    cufftHandle plan = 0;
    rc                = cufftCreate(&plan);
    assert(rc == CUFFT_SUCCESS);
    rc = cufftPlan2d(&plan, // plan handle
                      Nx, // transform length
                      Ny, // transform length
                      CUFFT_D2Z); // transform type (CUFFT_R2C for single-precision)
    assert(rc == CUFFT_SUCCESS);

    // Execute plan:
    // cufftExecD2Z: double precision, cufftExecR2C: single-precision
    rc = cufftExecD2Z(plan, x, (cufftDoubleComplex*)y);
    assert(rc == CUFFT_SUCCESS);

    // copy the output data to the host:
    std::vector<std::complex<double>> cdata(Nx * Ny);
    cudaMemcpy(cdata.data(), y, complex_bytes, cudaMemcpyDeviceToHost);
    std::cout << "output (abs):\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; j++)
        {
            auto pos = i * Nycomplex + j;
            std::cout << std::abs(cdata[pos]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Create reverse plan:
    rc    = CUFFT_SUCCESS;
    cufftHandle plan2;
    int n[2] = { Nx, Ny };
    int inembed[2] = { n[0], n[1] / 2 + 1 };
    int onembed[2] = { n[0], n[1] };
    rc = cufftPlanMany(&plan2, 2, n,
                       nullptr, 1, Nx*Nycomplex,
                       nullptr, 1, Nx*Ny,
                       CUFFT_Z2D, 1);
    assert(rc == CUFFT_SUCCESS);

    // Execute plan inverse
    rc = cufftExecZ2D(plan2, (cufftDoubleComplex*)y, x2);
    assert(rc == CUFFT_SUCCESS);

    rt = cudaMemcpy(rdata.data(), x2, real_bytes, cudaMemcpyDeviceToHost);
    assert(rt == cudaSuccess);

    std::cout << "roundtrip:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * rstride + j;
            std::cout << std::fixed << std::setprecision(2)
                      << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaFree(x);
    cudaFree(x2);
    cudaFree(y);

    return 0;
}
