# Zero-copy NPU<->GPU<->host substrate + real parallel proof.
#
# Build:  make test_parallel_real     # real NPU+GPU parallel compute proof
#         make                        # libsharedbo.a + substrate proofs
# Run:    sudo ./test_parallel_real   # need accel + render access
#         ./test_zero_copy            # substrate proof
#         ./test_pipeline             # pipeline timing
#
# Needs: XRT (2.21+), ROCm HIP (7.2+), /dev/accel/accel0.

ROCM   ?= /opt/rocm-7.2.4
XRTINC ?= /usr/include/xrt
XRTLIB ?= /usr/lib/x86_64-linux-gnu
CXX     := /opt/rocm-7.2.4/lib/llvm/bin/amdclang++

CXXFLAGS = -std=c++20 -O2 -g -Wall -Wno-unused-parameter -Wno-unused-result -Wno-unused-value \
           -D__HIP_PLATFORM_AMD__ \
           -I. -I$(XRTINC) -I$(ROCM)/include
LDFLAGS  = -L$(XRTLIB) -L$(ROCM)/lib \
           -lxrt_coreutil -luuid -ldl -lpthread -lamdhip64

.PHONY: all clean

all: libsharedbo.a test_zero_copy test_pipeline

libsharedbo.a: shared_bo.o
	ar rcs $@ $^

# ---- HIP-compiled substrate proofs (C++ only, HIP runtime linked) ----
shared_bo.o: shared_bo.cpp shared_bo.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_zero_copy: test_zero_copy.cpp shared_bo.o
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS)

test_pipeline: test_pipeline.cpp pipeline_overlap.cpp pipeline_overlap.h shared_bo.o
	$(CXX) $(CXXFLAGS) $< pipeline_overlap.cpp shared_bo.o -o $@ $(LDFLAGS)

# ---- TRUE parallel proof: HIP __global__ kernel + NPU xclbin on same pages ----
test_parallel_real: test_parallel_real.hip shared_bo.o shared_bo.h
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.a test_zero_copy test_pipeline test_parallel_real
