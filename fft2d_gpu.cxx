#include <CL/sycl.hpp>
#include <complex>
#include <iostream>
#include <oneapi/mkl.hpp>

void fft(cl::sycl::queue &q, double *datain, std::complex<double> *dataout,
         double *datainout, int sz) {
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                 oneapi::mkl::dft::domain::REAL>
        handle({sz, sz});
    MKL_LONG strides[3];
    strides[0] = 0;
    strides[1] = 1;
    strides[2] = sz;

    double *dataout2 = reinterpret_cast<double *>(dataout);

    //handle.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 1);
    handle.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                     DFTI_NOT_INPLACE);
    handle.set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, 1.0);
    handle.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0);
    handle.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                     DFTI_COMPLEX_COMPLEX);
    handle.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
    handle.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides);
    //handle.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, sz*sz);
    //handle.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, sz*(sz/2+1));
    handle.commit(q);
    auto e = oneapi::mkl::dft::compute_forward(handle, datain, dataout2);
    e.wait();
    auto e2 = oneapi::mkl::dft::compute_backward(handle, dataout2, datainout);
    e2.wait();
}

int main(int argc, char *argv[]) {
    cl::sycl::queue q{};
    int size = 17;
    double *in = cl::sycl::malloc_shared<double>(size * size, q);
    std::complex<double> *out =
        cl::sycl::malloc_shared<std::complex<double>>(size * (size / 2 + 1), q);
    double *inout = cl::sycl::malloc_shared<double>(size * size, q);

    q.memset(in, 0, size * size * sizeof(*in));
    q.wait();
    in[8 * size + 8] = 1.0;

    std::cout << "in[0,0]   : " << in[0] << std::endl;
    std::cout << "in[8,8]   : " << in[8 * size + 8] << std::endl;
    std::cout << "inout[0,0]: " << inout[0] << std::endl;
    std::cout << "inout[0,1]: " << inout[1] << std::endl;
    std::cout << "inout[8,8]: " << inout[8 * size + 8] << std::endl;

    fft(q, in, out, inout, size);

    std::cout << "after fft" << std::endl;
    std::cout << "out[0,0]  : " << out[0] << std::endl;
    std::cout << "out[0,1]  : " << out[1] << std::endl;
    std::cout << "out[8,8]  : " << out[8 * size + 8] << std::endl;
    std::cout << "inout[0,0]: " << inout[0] << std::endl;
    std::cout << "inout[0,1]: " << inout[1] << std::endl;
    std::cout << "inout[8,8]: " << inout[8 * size + 8] << std::endl;
    cl::sycl::free(in, q);
    cl::sycl::free(out, q);
    cl::sycl::free(inout, q);
    return 0;
}
