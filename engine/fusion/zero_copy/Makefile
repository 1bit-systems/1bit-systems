# Zero-copy NPU<->GPU<->host substrate.
#
# Builds the SharedBO library + a self-contained zero-copy proof.
# Needs: XRT (2.21+), ROCm HIP (7.2+). Run as root or with renderD128/accel0 access.
#
#   make            # build libsharedbo.a + test_zero_copy
#   sudo ./test_zero_copy

ROCM   ?= /opt/rocm-7.2.4
XRTINC ?= /usr/include/xrt
XRTLIB ?= /usr/lib/x86_64-linux-gnu

CXX     ?= g++
CXXFLAGS = -std=c++20 -O2 -g -Wall -Wno-unused-parameter \
           -D__HIP_PLATFORM_AMD__ \
           -I. -I$(XRTINC) -I$(ROCM)/include
LDFLAGS  = -L$(XRTLIB) -L$(ROCM)/lib \
           -lxrt_coreutil -luuid -ldl -lpthread \
           -lamdhip64

all: libsharedbo.a test_zero_copy test_pipeline

libsharedbo.a: shared_bo.o
	ar rcs $@ $^

shared_bo.o: shared_bo.cpp shared_bo.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_zero_copy: test_zero_copy.cpp shared_bo.o shared_bo.h
	$(CXX) $(CXXFLAGS) $< shared_bo.o -o $@ $(LDFLAGS)

test_pipeline: test_pipeline.cpp pipeline_overlap.cpp pipeline_overlap.h shared_bo.o
	$(CXX) $(CXXFLAGS) $< pipeline_overlap.cpp shared_bo.o -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.a test_zero_copy test_pipeline

.PHONY: all clean
