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

#include <hip/hip_runtime.h>
#include <hipfft.h>

int main()
{
    std::cout << "hipfft 2D double-precision real-to-complex transform\n";

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
    hipError_t rt;
    rt = hipMalloc(&x, real_bytes);
    assert(rt == HIP_SUCCESS);
    rt = hipMalloc(&x2, real_bytes);
    assert(rt == HIP_SUCCESS);
    rt = hipMalloc(&y, complex_bytes);
    assert(rt == HIP_SUCCESS);

    std::vector<double> rdata(Nx * rstride);
    // 2d delta function, zero except for one point that is 1.0
    rdata[Nx * 8 + 8] = 1.0;

    rt = hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);
    assert(rt == HIP_SUCCESS);

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
    hipfftResult rc   = HIPFFT_SUCCESS;
    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan2d(&plan, // plan handle
                      Nx, // transform length
                      Ny, // transform length
                      HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precision)
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // hipfftExecD2Z: double precision, hipfftExecR2C: single-precision
    rc = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)y);
    assert(rc == HIPFFT_SUCCESS);

    // copy the output data to the host:
    std::vector<std::complex<double>> cdata(Nx * Ny);
    hipMemcpy(cdata.data(), y, complex_bytes, hipMemcpyDeviceToHost);
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
    rc    = HIPFFT_SUCCESS;
    hipfftHandle plan2 = NULL;
    rc                 = hipfftCreate(&plan2);
    assert(rc == HIPFFT_SUCCESS);
    int n[2] = { Nx, Ny };
    int inembed[2] = { n[0], n[1] / 2 + 1 };
    /* This works correctly
     rc = hipfftPlanMany(&plan2, 2, n,
                        inembed, 1, Nx*Nycomplex,
                        n, 1, Nx*Ny,
                        HIPFFT_Z2D, 1);
    */
    // produces some garbage values in result
    rc = hipfftPlanMany(&plan2, 2, n,
                        nullptr, 1, Nx*Nycomplex,
                        nullptr, 1, Nx*Ny,
                        HIPFFT_Z2D, 1);
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan inverse
    rc = hipfftExecZ2D(plan2, (hipfftDoubleComplex*)y, x2);
    assert(rc == HIPFFT_SUCCESS);

    rt = hipMemcpy(rdata.data(), x2, real_bytes, hipMemcpyDeviceToHost);
    assert(rt == HIP_SUCCESS);

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

    hipfftDestroy(plan);
    hipfftDestroy(plan2);
    hipFree(x);
    hipFree(x2);
    hipFree(y);

    return 0;
}
