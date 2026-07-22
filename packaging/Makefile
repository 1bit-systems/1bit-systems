# 1bit.systems — build & package. Version is read from the root VERSION file
# (single source of truth — see issue #117); do not hardcode it here.
# One binary. 5 models. 120KB. Zero deps.
VERSION := $(shell [ -f ../VERSION ] && tr -d '[:space:]' < ../VERSION || echo "0.0.0")
.PHONY: all npu gpu package-deb package-aur package-brew clean

all: npu

npu:
	mkdir -p engine/npu/build
	gcc -c -std=c11 -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c
	g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math \
		-o engine/npu/build/npu_engine_all \
		engine/npu/src/npu_engine_all.cpp \
		engine/npu/build/dequant_q4nx.o \
		-Iengine/npu/src \
		-I$(XRT)/include -L$(XRT)/lib64 \
		-lxrt_coreutil -luuid -lm -ldl

# ZINC_DIR defaults to a sibling checkout; override on the command line, e.g.
#   make gpu ZINC_DIR=/path/to/zinc      (issue #152 — no hardcoded $HOME)
ZINC_DIR ?= $(HOME)/zinc
gpu:
	cd "$(ZINC_DIR)" && zig build -Doptimize=ReleaseFast

package-deb: npu
	mkdir -p packaging/deb/usr/bin packaging/deb/usr/lib/1bit
	cp engine/npu/build/npu_engine_all packaging/deb/usr/bin/1bit-npu
	cp engine/npu/build/dequant_q4nx.o packaging/deb/usr/lib/1bit/
	cp packaging/binary/1bit-server packaging/deb/usr/bin/1bit-server
	dpkg-deb --build packaging/deb 1bit-systems.deb

package-snap: npu
	cp engine/npu/build/npu_engine_all .
	cp packaging/binary/1bit-server .
	cp engine/npu/build/dequant_q4nx.o .
	snapcraft

package-aur:
	cp packaging/aur/PKGBUILD .
	makepkg -si

package-brew:
	cp packaging/homebrew/1bit-systems.rb /opt/homebrew/Library/Taps/homebrew/homebrew-core/Formula/

package-docker:
	docker build -t 1bit-systems -f packaging/docker/Dockerfile .

package-tarball: npu
	mkdir -p 1bit-systems-$(VERSION)/bin 1bit-systems-$(VERSION)/lib
	cp engine/npu/build/npu_engine_all 1bit-systems-$(VERSION)/bin/1bit-npu
	cp engine/npu/build/dequant_q4nx.o 1bit-systems-$(VERSION)/lib/
	tar czf 1bit-systems-$(VERSION)-linux-amd64.tar.gz 1bit-systems-$(VERSION)/

clean:
	rm -f engine/npu/build/npu_engine_all engine/npu/build/dequant_q4nx.o
