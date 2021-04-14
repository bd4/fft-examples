MKLROOT=/opt/intel/mkl
LINK_OPTIONS=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
CPPFLAGS= -DMKL_ILP64  -m64  -I"${MKLROOT}/include"
ROCMROOT=/opt/rocm

.PHONY: all
all: fft2d fft2d_gpu

fft2d: fft2d.cxx
	g++ -o fft2d $(CPPFLAGS) $(LINK_OPTIONS) fft2d.cxx

fft2d_gpu: fft2d_gpu.cxx
	dpcpp -g -o fft2d_gpu -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core fft2d_gpu.cxx

fft2d_hip: fft2d_hip.cxx
	hipcc -g -o fft2d_hip -L$(ROCMROOT)/hipfft/lib -lhipfft -I$(ROCMROOT)/hipfft/include fft2d_hip.cxx

fft2d_cuda: fft2d_cuda.cxx
	nvcc -g -o fft2d_cuda -lcufft fft2d_cuda.cxx
