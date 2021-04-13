#include <complex>
#include <iostream>

// Based on example at https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Crash-for-a-simple-2D-MKL-FFT-example-when-matrix-size-is-not/td-p/1219066

#include "mkl.h"
#include "mkl_dfti.h"

void fft(double *datain, std::complex<double> *dataout, double *datainout,
         int sz) {
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG status;
    MKL_LONG sizes[2] = {sz, sz};
    MKL_LONG strides[3];
    strides[0] = 0;
    strides[1] = 1;
    strides[2] = sz;
    status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 2, sizes);
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(handle, DFTI_FORWARD_SCALE, 1.0);
    status = DftiSetValue(handle, DFTI_BACKWARD_SCALE, 1.0);
    status =
        DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(handle, DFTI_INPUT_STRIDES, strides);
    status = DftiSetValue(handle, DFTI_OUTPUT_STRIDES, strides);
    status = DftiCommitDescriptor(handle);
    status = DftiComputeForward(handle, datain, dataout);
    status = DftiComputeBackward(handle, dataout, datainout);
    status = DftiFreeDescriptor(&handle);
}

int main(int argc, char *argv[]) {
    int size = 17;
    std::cout << "2d fft of delta function, size = "
              << size << " by " << size << std::endl;
    double *in = (double *)mkl_calloc(size * size, sizeof(double), 64);
    std::complex<double> *out = (std::complex<double> *)mkl_calloc(
        size * (size / 2 + 1), sizeof(std::complex<double>), 64);
    double *inout = (double *)mkl_calloc(size * size, sizeof(double), 64);
    in[8 * size + 8] = 1.0;
    fft(in, out, inout, size);

    std::cout << "inout[0,0]: " << inout[0]
              << " (near 0 expected)" << std::endl;
    std::cout << "inout[8,8]: " << inout[8 * size + 8]
              << " (289 = 1 * size^2 expected)" << std::endl;
    mkl_free(in);
    mkl_free(out);
    mkl_free(inout);
    return 0;
}
