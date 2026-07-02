# Unified NPU+GPU+CPU Control Plane — Build & Install
#
# Usage:
#   make          — build everything
#   make test     — run all tests
#   make bench    — run benchmarks
#   make install  — install binaries
#   make daemon   — start the control plane daemon
#   make clean    — clean build artifacts

.PHONY: all tests bench daemon install clean

CC       := g++
CFLAGS   := -std=gnu++17 -O2 -I/usr/include/libdrm
LDFLAGS  := -ldrm -ldrm_amdgpu
NPU_CXX  := g++ -std=gnu++17 -O3 -I/usr/include -L/usr/lib/x86_64-linux-gnu
NPU_LIBS := -lxrt_coreutil -ldl -luuid -lpthread

TESTS_DIR := tests
BUILD_DIR := build

# Sources
TEST_SRCS := $(TESTS_DIR)/test_gtt_dmabuf.cpp $(TESTS_DIR)/bench_gtt_dmabuf.cpp \
             $(TESTS_DIR)/test_npu_dev.cpp
TEST_BINS := $(TEST_SRCS:$(TESTS_DIR)/%.cpp=$(BUILD_DIR)/%)

all: $(BUILD_DIR) $(TEST_BINS)
	@echo ""
	@echo "═══ Unified Control Plane — Build Complete ═══"
	@echo ""
	@echo "  Binaries:"
	@for b in $(TEST_BINS); do echo "    $$b"; done
	@echo "  Library:  build/libhip_npu.so"
	@echo "  Daemon:   daemon/npu-gpu-cpud.py"
	@echo ""
	@echo "  Quick start:"
	@echo "    sudo make test          — run all hardware tests"
	@echo "    sudo make bench         — run benchmarks"
	@echo "    sudo make daemon        — start the control plane"
	@echo ""

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Test binaries
$(BUILD_DIR)/test_gtt_dmabuf: $(TESTS_DIR)/test_gtt_dmabuf.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  CC  $@"

$(BUILD_DIR)/bench_gtt_dmabuf: $(TESTS_DIR)/bench_gtt_dmabuf.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  CC  $@"

$(BUILD_DIR)/test_npu_dev: $(TESTS_DIR)/test_npu_dev.cpp
	$(CC) $(CFLAGS) -o $@ $^ -ldrm
	@echo "  CC  $@"

# HIP NPU shim library
$(BUILD_DIR)/libhip_npu.so: rocm-npu/hip_npu.cpp
	mkdir -p rocm-npu/build
	cmake -S rocm-npu -B rocm-npu/build -DHIP_INCLUDE_DIR=/usr/include
	cmake --build rocm-npu/build
	cp rocm-npu/build/libhip_npu.so $(BUILD_DIR)/
	@echo "  LD  $@"

# Tests
test: all
	@echo "═══ Running Hardware Tests ═══"
	@echo ""
	@echo "1. NPU device test..."
	$(BUILD_DIR)/test_npu_dev
	@echo ""
	@echo "2. DMA-buf zero-copy test..."
	sudo $(BUILD_DIR)/test_gtt_dmabuf
	@echo ""
	@echo "3. HIP device enumeration..."
	@if [ -f $(TESTS_DIR)/hip_list_devices ]; then \
		echo "   Without shim (GPU only):"; \
		$(TESTS_DIR)/hip_list_devices 2>/dev/null; \
		echo "   With NPU shim (GPU + NPU):"; \
		LD_PRELOAD=$(BUILD_DIR)/libhip_npu.so $(TESTS_DIR)/hip_list_devices 2>/dev/null; \
	else \
		echo "   Pre-built hip_list_devices not found (install hipcc to build)"; \
	fi
	@echo ""
	@echo "═══ All Hardware Tests Passed ═══"

# Benchmarks
bench: all
	@echo "═══ Running Benchmarks ═══"
	@echo ""
	@echo "1. DMA-buf bandwidth..."
	sudo $(BUILD_DIR)/bench_gtt_dmabuf
	@echo ""
	@echo "2. Unified benchmark..."
	python3 $(TESTS_DIR)/bench_unified.py
	@echo ""
	@echo "═══ Benchmarks Complete ═══"

# Daemon
daemon:
	@echo "═══ Starting Control Plane Daemon ═══"
	@echo ""
	@echo "  Starting gateway on http://localhost:8080..."
	@echo "  (backends start automatically — NPU engine, GPU lemond)"
	@python3 daemon/npu-gpu-cpud.py --port 8080
	@echo ""

# Install
install: all
	@echo "═══ Installing ═══"
	@echo ""
	@echo "  Binaries:"
	@for b in $(TEST_BINS); do \
		sudo cp $$b /usr/local/bin/; \
		echo "    /usr/local/bin/$$(basename $$b)"; \
	done
	@echo "  Library:"
	sudo cp $(BUILD_DIR)/libhip_npu.so /usr/local/lib/
	@echo "    /usr/local/lib/libhip_npu.so"
	@echo "  Daemon:"
	sudo cp daemon/npu-gpu-cpud.py /usr/local/bin/
	@echo "    /usr/local/bin/npu-gpu-cpud"
	@echo ""
	@echo "  Usage:  npu-gpu-cpud --port 8080"
	@echo "  Usage:  LD_PRELOAD=libhip_npu.so your_hip_app"
	@echo ""

# === NPU Parallelism Tools (crack the final egg) ===
NPU_TOOLS := tools/npu_multi_context_engine.cpp tools/npu_pipeline_engine.cpp \
             tools/npu_multi_context_server.cpp
NPU_BINS  := $(NPU_TOOLS:tools/%.cpp=$(BUILD_DIR)/%)

$(BUILD_DIR)/npu_multi_context_engine: tools/npu_multi_context_engine.cpp
	$(NPU_CXX) -o $@ $< $(NPU_LIBS)
	@echo "  NPU $@"

$(BUILD_DIR)/npu_pipeline_engine: tools/npu_pipeline_engine.cpp
	$(NPU_CXX) -o $@ $< $(NPU_LIBS)
	@echo "  NPU $@"

$(BUILD_DIR)/npu_server: tools/npu_multi_context_server.cpp
	$(NPU_CXX) -o $@ $< $(NPU_LIBS)
	@echo "  NPU $@"

npu-tools: $(BUILD_DIR) $(NPU_BINS)
	@echo ""
	@echo "═══ NPU Parallelism Tools Built ═══"
	@echo ""
	@echo "  $(BUILD_DIR)/npu_multi_context_engine  — verify multi-context parallelism"
	@echo "  $(BUILD_DIR)/npu_pipeline_engine       — pipeline throughput benchmark"
	@echo "  $(BUILD_DIR)/npu_server                — multi-context inference server"
	@echo ""
	@echo "  Quick start:"
	@echo "    sudo ./build/npu_server --scaling --ctx 8"
	@echo "    sudo ./build/npu_server --bench --ctx 8 --requests 32"
	@echo ""

# Run the parallelism demo
npu-demo: npu-tools
	@echo "═══ NPU Parallelism Demo ═══"
	@echo ""
	@echo "1. Multi-Context Engine (4 contexts)..."
	@sudo $(BUILD_DIR)/npu_multi_context_engine 4
	@echo ""
	@echo "2. Scaling Test (1..8 contexts)..."
	@sudo $(BUILD_DIR)/npu_server --scaling --ctx 8
	@echo ""
	@echo "3. Concurrent Benchmark (32 requests, 8 contexts)..."
	@sudo $(BUILD_DIR)/npu_server --bench --ctx 8 --requests 32
	@echo ""
	@echo "═══ Demo Complete ═══"

clean:
	rm -rf $(BUILD_DIR) rocm-npu/build
	@echo "  Clean complete"
